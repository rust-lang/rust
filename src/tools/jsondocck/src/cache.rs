use std::collections::HashMap;
use std::path::Path;

use fs_err as fs;
use jaq_core::load::{Arena, File, Loader};
use jaq_core::{Ctx, Vars, data, unwrap_valr};
use jaq_json::Val;
use serde_json::Value;

use crate::config::Config;

#[derive(Debug)]
pub struct Cache {
    jq_value: Val,
    pub jq_variables: HashMap<String, Val>,
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
            jq_value: serde_json::from_str::<Val>(&content).expect("failed to convert from JSON"),
            jq_variables: HashMap::new(),
            value: serde_json::from_str::<Value>(&content).expect("failed to convert from JSON"),
            variables: HashMap::from([("FILE".to_owned(), config.template.clone().into())]),
        }
    }

    // FIXME: Make this failible, so jsonpath syntax error has line number.
    pub fn select(&self, path: &str) -> Vec<&Value> {
        jsonpath_rust::query::js_path_vals(path, &self.value).unwrap()
    }

    pub fn jq_select(&self, path: &str) -> Vec<Val> {
        let program = File { code: path, path: () };

        let defs = jaq_core::defs().chain(jaq_std::defs()).chain(jaq_json::defs());
        let funs = jaq_core::funs().chain(jaq_std::funs()).chain(jaq_json::funs());

        let loader = Loader::new(defs);
        let arena = Arena::default();

        let modules = loader.load(&arena, program).unwrap();

        let vars = self.jq_variables.iter().map(|t| (t.0, t.1));

        let filter = jaq_core::Compiler::default()
            .with_funs(funs)
            .with_global_vars(vars.clone().map(|t| t.0.as_str()))
            .compile(modules)
            .unwrap();

        let ctx = Ctx::<data::JustLut<Val>>::new(&filter.lut, Vars::new(vars.map(|t| t.1.clone())));

        filter.id.run((ctx, self.jq_value.clone())).map(unwrap_valr).map(Result::unwrap).collect()
    }
}
