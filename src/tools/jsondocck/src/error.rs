use crate::Command;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum CkError {
    /// A check failed. File didn't exist or failed to match the command
    FailedCheck(String, Command),
    /// An error triggered by some other error
    Induced(Box<dyn Error>),
}

impl fmt::Display for CkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CkError::FailedCheck(msg, cmd) => {
                write!(f, "Failed check: {} on line {}", msg, cmd.lineno)
            }
            CkError::Induced(err) => write!(f, "Check failed: {}", err),
        }
    }
}

impl<T: Error + 'static> From<T> for CkError {
    fn from(err: T) -> CkError {
        CkError::Induced(Box::new(err))
    }
}
