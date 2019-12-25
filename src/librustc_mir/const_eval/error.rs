use std::error::Error;
use std::fmt;

use crate::interpret::InterpErrorInfo;

#[derive(Clone, Debug)]
pub enum ConstEvalError {
    NeedsRfc(String),
    ConstAccessesStatic,
}

impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalError {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_unsup!(Unsupported(self.to_string())).into()
    }
}

impl fmt::Display for ConstEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(f, "\"{}\" needs an rfc before being allowed inside constants", msg)
            }
            ConstAccessesStatic => write!(f, "constant accesses static"),
        }
    }
}

impl Error for ConstEvalError {}
