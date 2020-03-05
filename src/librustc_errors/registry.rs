use rustc_data_structures::fx::FxHashMap;

#[derive(Debug)]
pub struct InvalidErrorCode;

#[derive(Clone)]
pub struct Registry {
    long_descriptions: FxHashMap<&'static str, Option<&'static str>>,
}

impl Registry {
    pub fn new(long_descriptions: &[(&'static str, Option<&'static str>)]) -> Registry {
        Registry { long_descriptions: long_descriptions.iter().cloned().collect() }
    }

    /// This will panic if an invalid error code is passed in
    pub fn find_description(&self, code: &str) -> Option<&'static str> {
        self.try_find_description(code).unwrap()
    }
    /// Returns `InvalidErrorCode` if the code requested does not exist in the
    /// registry. Otherwise, returns an `Option` where `None` means the error
    /// code is valid but has no extended information.
    pub fn try_find_description(
        &self,
        code: &str,
    ) -> Result<Option<&'static str>, InvalidErrorCode> {
        if !self.long_descriptions.contains_key(code) {
            return Err(InvalidErrorCode);
        }
        Ok(*self.long_descriptions.get(code).unwrap())
    }
}
