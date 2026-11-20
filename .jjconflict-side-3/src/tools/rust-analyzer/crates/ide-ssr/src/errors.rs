//! Code relating to errors produced by SSR.

/// Constructs an SsrError taking arguments like the format macro.
macro_rules! _error {
    ($fmt:expr) => {$crate::SsrError::new(format!($fmt))};
    ($fmt:expr, $($arg:tt)+) => {$crate::SsrError::new(format!($fmt, $($arg)+))}
}
pub(crate) use _error as error;

/// Returns from the current function with an error, supplied by arguments as for format!
macro_rules! _bail {
    ($($tokens:tt)*) => {return Err(crate::errors::error!($($tokens)*))}
}
pub(crate) use _bail as bail;

#[derive(Debug, PartialEq)]
pub struct SsrError(pub(crate) String);

impl std::fmt::Display for SsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Parse error: {}", self.0)
    }
}

impl SsrError {
    pub(crate) fn new(message: impl Into<String>) -> SsrError {
        SsrError(message.into())
    }
}
