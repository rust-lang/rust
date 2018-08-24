use std::collections::HashMap;

#[derive(Clone)]
pub struct Registry {
    descriptions: HashMap<&'static str, &'static str>,
}

impl Registry {
    pub fn new(descriptions: &[(&'static str, &'static str)]) -> Registry {
        Registry { descriptions: descriptions.iter().cloned().collect() }
    }

    pub fn find_description(&self, code: &str) -> Option<&'static str> {
        self.descriptions.get(code).cloned()
    }
}
