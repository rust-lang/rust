use rustc_data_structures::fx::FxHashMap;

use crate::ErrCode;

#[derive(Debug)]
pub struct InvalidErrorCode;

#[derive(Clone)]
pub struct Registry {
    long_descriptions: FxHashMap<ErrCode, &'static str>,
}

impl Registry {
    pub fn new(long_descriptions: &[(ErrCode, &'static str)]) -> Registry {
        Registry { long_descriptions: long_descriptions.iter().copied().collect() }
    }

    /// Returns `InvalidErrorCode` if the code requested does not exist in the
    /// registry.
    pub fn try_find_description(&self, code: ErrCode) -> Result<&'static str, InvalidErrorCode> {
        self.long_descriptions.get(&code).copied().ok_or(InvalidErrorCode)
    }
}
