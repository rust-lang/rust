use std::error::Error;
use std::fmt;

use rustc_middle::mir::interpret::ConstEvalErr;
use rustc_middle::mir::AssertKind;
use rustc_middle::ty::ConstInt;
use rustc_span::{Span, Symbol};

use super::InterpCx;
use crate::interpret::{InterpErrorInfo, Machine, MachineStopType};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    NeedsRfc(String),
    ConstAccessesStatic,
    ModifiedGlobal,
    AssertFailure(AssertKind<ConstInt>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
    Abort(String),
}

impl MachineStopType for ConstEvalErrKind {
    fn is_hard_err(&self) -> bool {
        matches!(self, Self::Panic { .. })
    }
}

// The errors become `MachineStop` with plain strings when being raised.
// `ConstEvalErr` (in `librustc_middle/mir/interpret/error.rs`) knows to
// handle these.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self).into()
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
            Abort(ref msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for ConstEvalErrKind {}

/// When const-evaluation errors, this type is constructed with the resulting information,
/// and then used to emit the error as a lint or hard error.
#[derive(Debug)]
pub struct ConstEvalError<'tcx> {
    inner: ConstEvalErr<'tcx>,
}

impl<'tcx> ConstEvalError<'tcx> {
    /// Turn an interpreter error into something to report to the user.
    /// As a side-effect, if RUSTC_CTFE_BACKTRACE is set, this prints the backtrace.
    /// Should be called only if the error is actually going to to be reported!
    pub fn new<'mir, M: Machine<'mir, 'tcx>>(
        ecx: &InterpCx<'mir, 'tcx, M>,
        error: InterpErrorInfo<'tcx>,
        span: Option<Span>,
    ) -> ConstEvalError<'tcx>
    where
        'tcx: 'mir,
    {
        error.print_backtrace();
        let stacktrace = ecx.generate_stacktrace();
        let inner = ConstEvalErr {
            error: error.into_kind(),
            stacktrace,
            span: span.unwrap_or_else(|| ecx.cur_span()),
        };

        ConstEvalError { inner }
    }

    pub fn into_inner(self) -> ConstEvalErr<'tcx> {
        self.inner
    }
}
