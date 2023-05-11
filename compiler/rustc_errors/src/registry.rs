use rustc_data_structures::fx::FxHashMap;

#[derive(Debug)]
pub struct InvalidErrorCode;

#[derive(Clone)]
pub struct Registry {
    long_descriptions: FxHashMap<&'static str, &'static str>,
}

impl Registry {
    pub fn new(long_descriptions: &[(&'static str, &'static str)]) -> Registry {
        Registry { long_descriptions: long_descriptions.iter().copied().collect() }
    }

    /// Returns `InvalidErrorCode` if the code requested does not exist in the
    /// registry.
    pub fn try_find_description(&self, code: &str) -> Result<&'static str, InvalidErrorCode> {
        self.long_descriptions.get(code).copied().ok_or(InvalidErrorCode)
    }
}
