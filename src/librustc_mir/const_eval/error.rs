use std::error::Error;
use std::fmt;

use rustc_middle::mir::AssertKind;
use rustc_span::Symbol;

use super::InterpCx;
use crate::interpret::{ConstEvalErr, InterpErrorInfo, Machine};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    NeedsRfc(String),
    ConstAccessesStatic,
    ModifiedGlobal,
    AssertFailure(AssertKind<u64>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
}

// The errors become `MachineStop` with plain strings when being raised.
// `ConstEvalErr` (in `librustc_middle/mir/interpret/error.rs`) knows to
// handle these.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self.to_string()).into()
    }
}

impl fmt::Display for ConstEvalErrKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::ConstEvalErrKind::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(f, "\"{}\" needs an rfc before being allowed inside constants", msg)
            }
            ConstAccessesStatic => write!(f, "constant accesses static"),
            ModifiedGlobal => {
                write!(f, "modifying a static's initial value from another static's initializer")
            }
            AssertFailure(ref msg) => write!(f, "{:?}", msg),
            Panic { msg, line, col, file } => {
                write!(f, "the evaluated program panicked at '{}', {}:{}:{}", msg, file, line, col)
            }
        }
    }
}

impl Error for ConstEvalErrKind {}

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
