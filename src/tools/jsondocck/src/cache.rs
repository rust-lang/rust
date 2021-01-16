use crate::error::CkError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::{fs, io};

#[derive(Debug)]
pub struct Cache {
    root: PathBuf,
    files: HashMap<PathBuf, String>,
    values: HashMap<PathBuf, Value>,
    last_path: Option<PathBuf>,
}

impl Cache {
    pub fn new(doc_dir: &str) -> Cache {
        Cache {
            root: Path::new(doc_dir).to_owned(),
            files: HashMap::new(),
            values: HashMap::new(),
            last_path: None,
        }
    }

    fn resolve_path(&mut self, path: &String) -> PathBuf {
        if path != "-" {
            let resolve = self.root.join(path);
            self.last_path = Some(resolve.clone());
            resolve
        } else {
            self.last_path.as_ref().unwrap().clone()
        }
    }

    pub fn get_file(&mut self, path: &String) -> Result<String, io::Error> {
        let path = self.resolve_path(path);

        if let Some(f) = self.files.get(&path) {
            return Ok(f.clone());
        }

        let file = fs::read_to_string(&path)?;

        self.files.insert(path, file.clone());

        Ok(file)
    }

    pub fn get_value(&mut self, path: &String) -> Result<Value, CkError> {
        let path = self.resolve_path(path);

        if let Some(v) = self.values.get(&path) {
            return Ok(v.clone());
        }

        let file = fs::File::open(&path)?;

        let val = serde_json::from_reader::<_, Value>(file)?;

        self.values.insert(path, val.clone());

        Ok(val)
    }
}
