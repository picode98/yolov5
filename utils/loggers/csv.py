from typing import Dict, Any


class CSVLogger:
    def __init__(self, path: str, delim: str = ','):
        self.file_ref = open(path, 'w', encoding='utf8')
        self.col_headers = None
        self.delim = delim

    def escape(self, value: str):
        if self.delim in value or '"' in value:
            return '"' + value.replace('"', '""') + '"'
        else:
            return value

    def log(self, data: Dict[str, Any]):
        if self.col_headers is None:
            self.col_headers = list(data.keys())
            self.file_ref.write(self.delim.join(self.escape(header) for header in self.col_headers) + '\n')
        elif set(data.keys()) != set(self.col_headers):
            raise ValueError(f'Inconsistent data format for CSV logging (expected keys to be {self.col_headers})')

        values = [str(data[key]) for key in self.col_headers]
        self.file_ref.write(self.delim.join(self.escape(value) for value in values) + '\n')

    def close(self):
        self.file_ref.close()