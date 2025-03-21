use std::collections::HashMap;
use std::path::Path;

use fs_err as fs;
use serde_json::Value;

use crate::config::Config;

#[derive(Debug)]
pub struct Cache {
    value: Value,
    pub variables: HashMap<String, Value>,
}

impl Cache {
    /// Create a new cache, used to read files only once and otherwise store their contents.
    pub fn new(config: &Config) -> Cache {
        let root = Path::new(&config.doc_dir);
        // `filename` needs to replace `-` with `_` to be sure the JSON path will always be valid.
        let filename =
            Path::new(&config.template).file_stem().unwrap().to_str().unwrap().replace('-', "_");
        let file_path = root.join(&Path::with_extension(Path::new(&filename), "json"));
        let content = fs::read_to_string(&file_path).expect("failed to read JSON file");

        Cache {
            value: serde_json::from_str::<Value>(&content).expect("failed to convert from JSON"),
            variables: HashMap::from([("FILE".to_owned(), config.template.clone().into())]),
        }
    }

    // FIXME: Make this failible, so jsonpath syntax error has line number.
    pub fn select(&self, path: &str) -> Vec<&Value> {
        jsonpath_rust::query::js_path_vals(path, &self.value).unwrap()
    }
}
