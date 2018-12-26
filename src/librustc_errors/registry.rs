use rustc_data_structures::fx::FxHashMap;

#[derive(Clone)]
pub struct Registry {
    descriptions: FxHashMap<&'static str, &'static str>,
}

impl Registry {
    pub fn new(descriptions: &[(&'static str, &'static str)]) -> Registry {
        Registry { descriptions: descriptions.iter().cloned().collect() }
    }

    pub fn find_description(&self, code: &str) -> Option<&'static str> {
        self.descriptions.get(code).cloned()
    }
}
