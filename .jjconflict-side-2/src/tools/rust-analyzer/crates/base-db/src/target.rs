//! Information about the target.

use std::fmt;

use triomphe::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    // Only what we need is present here.
    Wasm32,
    Wasm64,
    Other,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct TargetData {
    pub data_layout: Box<str>,
    pub arch: Arch,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TargetLoadError(Arc<str>);

impl fmt::Debug for TargetLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for TargetLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for TargetLoadError {}

impl From<String> for TargetLoadError {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<&str> for TargetLoadError {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

pub type TargetLoadResult = Result<TargetData, TargetLoadError>;
