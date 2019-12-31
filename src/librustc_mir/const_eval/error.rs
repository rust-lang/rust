use std::error::Error;
use std::fmt;

use super::InterpCx;
use crate::interpret::{ConstEvalErr, InterpErrorInfo, Machine};
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

/// Turn an interpreter error into something to report to the user.
/// As a side-effect, if RUSTC_CTFE_BACKTRACE is set, this prints the backtrace.
/// Should be called only if the error is actually going to to be reported!
pub fn error_to_const_error<'mir, 'tcx, M: Machine<'mir, 'tcx>>(
    ecx: &InterpCx<'mir, 'tcx, M>,
    mut error: InterpErrorInfo<'tcx>,
) -> ConstEvalErr<'tcx> {
    error.print_backtrace();
    let stacktrace = ecx.generate_stacktrace(None);
    ConstEvalErr { error: error.kind, stacktrace, span: ecx.tcx.span }
}
