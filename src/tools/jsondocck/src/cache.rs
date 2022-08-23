use crate::config::Config;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use fs_err as fs;

#[derive(Debug)]
pub struct Cache {
    value: Value,
    pub variables: HashMap<String, Value>,
}

impl Cache {
    /// Create a new cache, used to read files only once and otherwise store their contents.
    pub fn new(config: &Config) -> Cache {
        let root = Path::new(&config.doc_dir);
        let filename = Path::new(&config.template).file_stem().unwrap();
        let file_path = root.join(&Path::with_extension(Path::new(filename), "json"));
        let content = fs::read_to_string(&file_path).expect("failed to read JSON file");

        Cache {
            value: serde_json::from_str::<Value>(&content).expect("failed to convert from JSON"),
            variables: HashMap::new(),
        }
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}
