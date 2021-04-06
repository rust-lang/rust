use rustc_data_structures::fx::FxHashMap;

#[derive(Debug)]
pub struct InvalidErrorCode;

#[derive(Clone)]
pub struct Registry {
    long_descriptions: FxHashMap<&'static str, Option<&'static str>>,
}

impl Registry {
    pub fn new(long_descriptions: &[(&'static str, Option<&'static str>)]) -> Registry {
        Registry { long_descriptions: long_descriptions.iter().copied().collect() }
    }

    /// Returns `InvalidErrorCode` if the code requested does not exist in the
    /// registry. Otherwise, returns an `Option` where `None` means the error
    /// code is valid but has no extended information.
    pub fn try_find_description(
        &self,
        code: &str,
    ) -> Result<Option<&'static str>, InvalidErrorCode> {
        self.long_descriptions.get(code).copied().ok_or(InvalidErrorCode)
    }
}
