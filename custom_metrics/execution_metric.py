import sqlite3
import os
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracyMetric(BaseMetric):

    async_mode = False

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False

    def measure(self, test_case: LLMTestCase) -> float:
        sql_gerada = test_case.actual_output
        sql_correta = test_case.expected_output
        
        if test_case.context is None or not test_case.context:
            self.reason = "Falha: 'db_id' não encontrado no context do caso de teste."
            self.score = 0.0
            return self.score
            
        db_id = test_case.context[0]

        db_path = os.path.join('database', db_id, f'{db_id}.sqlite')

        if not os.path.exists(db_path):
            self.reason = f"Falha: Banco de dados '{db_id}' não encontrado em '{os.path.abspath(db_path)}'."
            self.score = 0.0
            return self.score

        try:
            conn = sqlite3.connect(db_path)
            cursor_gerada = conn.cursor()
            cursor_correta = conn.cursor()

            cursor_gerada.execute(sql_gerada)
            resultado_gerado = cursor_gerada.fetchall()

            cursor_correta.execute(sql_correta)
            resultado_correto = cursor_correta.fetchall()
            
            conn.close()

            if set(map(tuple, resultado_gerado)) == set(map(tuple, resultado_correto)):
                self.reason = "Sucesso: Os resultados da execução são idênticos."
                self.score = 1.0
            else:
                self.reason = f"Falha: Os resultados da execução são diferentes. Gerado={resultado_gerado}, Correto={resultado_correto}"
                self.score = 0.0

        except sqlite3.Error as e:
            self.reason = f"Falha: Erro de execução na SQL ('{sql_gerada}') - {e}"
            self.score = 0.0
        
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"