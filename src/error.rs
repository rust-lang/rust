use std::error::Error;
use std::fmt;

#[derive(Clone, Debug)]
pub enum EvalError {
    DanglingPointerDeref,
    InvalidBool,
    PointerOutOfBounds,
    InvalidPointerAccess,
}

pub type EvalResult<T> = Result<T, EvalError>;

impl Error for EvalError {
    fn description(&self) -> &str {
        match *self {
            EvalError::DanglingPointerDeref => "dangling pointer was dereferenced",
            EvalError::InvalidBool => "invalid boolean value read",
            EvalError::PointerOutOfBounds => "pointer offset outside bounds of allocation",
            EvalError::InvalidPointerAccess =>
                "a raw memory access tried to access part of a pointer value as bytes",
        }
    }

    fn cause(&self) -> Option<&Error> { None }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}
