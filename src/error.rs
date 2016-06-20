use std::error::Error;
use std::fmt;
use rustc::mir::repr as mir;
use rustc::ty::BareFnTy;
use memory::Pointer;
use rustc_const_math::ConstMathErr;
use syntax::codemap::Span;

#[derive(Clone, Debug)]
pub enum EvalError<'tcx> {
    FunctionPointerTyMismatch(&'tcx BareFnTy<'tcx>, &'tcx BareFnTy<'tcx>),
    DanglingPointerDeref,
    InvalidFunctionPointer,
    InvalidBool,
    InvalidDiscriminant,
    PointerOutOfBounds {
        ptr: Pointer,
        size: usize,
        allocation_size: usize,
    },
    ReadPointerAsBytes,
    ReadBytesAsPointer,
    InvalidPointerMath,
    ReadUndefBytes,
    InvalidBoolOp(mir::BinOp),
    Unimplemented(String),
    DerefFunctionPointer,
    ExecuteMemory,
    ArrayIndexOutOfBounds(Span, u64, u64),
    Math(Span, ConstMathErr),
    InvalidChar(u32),
}

pub type EvalResult<'tcx, T> = Result<T, EvalError<'tcx>>;

impl<'tcx> Error for EvalError<'tcx> {
    fn description(&self) -> &str {
        match *self {
            EvalError::FunctionPointerTyMismatch(..) =>
                "tried to call a function through a function pointer of a different type",
            EvalError::DanglingPointerDeref =>
                "dangling pointer was dereferenced",
            EvalError::InvalidFunctionPointer =>
                "tried to use a pointer as a function pointer",
            EvalError::InvalidBool =>
                "invalid boolean value read",
            EvalError::InvalidDiscriminant =>
                "invalid enum discriminant value read",
            EvalError::PointerOutOfBounds { .. } =>
                "pointer offset outside bounds of allocation",
            EvalError::ReadPointerAsBytes =>
                "a raw memory access tried to access part of a pointer value as raw bytes",
            EvalError::ReadBytesAsPointer =>
                "attempted to interpret some raw bytes as a pointer address",
            EvalError::InvalidPointerMath =>
                "attempted to do math or a comparison on pointers into different allocations",
            EvalError::ReadUndefBytes =>
                "attempted to read undefined bytes",
            EvalError::InvalidBoolOp(_) =>
                "invalid boolean operation",
            EvalError::Unimplemented(ref msg) => msg,
            EvalError::DerefFunctionPointer =>
                "tried to dereference a function pointer",
            EvalError::ExecuteMemory =>
                "tried to treat a memory pointer as a function pointer",
            EvalError::ArrayIndexOutOfBounds(..) =>
                "array index out of bounds",
            EvalError::Math(..) =>
                "mathematical operation failed",
            EvalError::InvalidChar(..) =>
                "tried to interpret an invalid 32-bit value as a char",
        }
    }

    fn cause(&self) -> Option<&Error> { None }
}

impl<'tcx> fmt::Display for EvalError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EvalError::PointerOutOfBounds { ptr, size, allocation_size } => {
                write!(f, "memory access of {}..{} outside bounds of allocation {} which has size {}",
                       ptr.offset, ptr.offset + size, ptr.alloc_id, allocation_size)
            },
            EvalError::FunctionPointerTyMismatch(expected, got) =>
                write!(f, "tried to call a function of type {:?} through a function pointer of type {:?}", expected, got),
            EvalError::ArrayIndexOutOfBounds(span, len, index) =>
                write!(f, "array index {} out of bounds {} at {:?}", index, len, span),
            EvalError::Math(span, ref err) =>
                write!(f, "mathematical operation at {:?} failed with {:?}", span, err),
            EvalError::InvalidChar(c) =>
                write!(f, "invalid utf8 character: {}", c),
            _ => write!(f, "{}", self.description()),
        }
    }
}
