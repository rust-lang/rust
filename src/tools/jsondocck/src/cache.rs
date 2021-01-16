use crate::error::CkError;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
            root: <str as AsRef<Path>>::as_ref(doc_dir).to_owned(),
            files: HashMap::new(),
            values: HashMap::new(),
            last_path: None,
        }
    }

    fn resolve_path(&mut self, path: &String) -> Result<PathBuf, CkError> {
        if path != "-" {
            let resolve = self.root.join(path);
            self.last_path = Some(resolve.clone());
            Ok(resolve)
        } else {
            match &self.last_path {
                Some(p) => Ok(p.clone()),
                None => unreachable!(),
            }
        }
    }

    pub fn get_file(&mut self, path: &String) -> Result<String, CkError> {
        let path = self.resolve_path(path)?;

        if let Some(f) = self.files.get(&path) {
            return Ok(f.clone());
        }

        let file = fs::read_to_string(&path)?;

        self.files.insert(path, file.clone());

        Ok(file)
        // Err(_) => Err(CkError::FailedCheck(format!("File {:?} does not exist / could not be opened", path)))
    }

    pub fn get_value(&mut self, path: &String) -> Result<Value, CkError> {
        let path = self.resolve_path(path)?;

        if let Some(v) = self.values.get(&path) {
            return Ok(v.clone());
        }

        let file = fs::File::open(&path)?;
        // Err(_) => return Err(CkError::FailedCheck(format!("File {:?} does not exist / could not be opened", path)))

        let val = serde_json::from_reader::<_, Value>(file)?;

        self.values.insert(path, val.clone());

        Ok(val)
        // Err(_) => Err(CkError::FailedCheck(format!("File {:?} did not contain valid JSON", path)))
    }
}
