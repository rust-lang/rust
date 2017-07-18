use std::error::Error;
use std::fmt;
use rustc::mir;
use rustc::ty::{FnSig, Ty, layout};
use memory::{MemoryPointer, LockInfo, AccessKind, Kind};
use rustc_const_math::ConstMathErr;
use syntax::codemap::Span;

#[derive(Clone, Debug)]
pub enum EvalError<'tcx> {
    FunctionPointerTyMismatch(FnSig<'tcx>, FnSig<'tcx>),
    NoMirFor(String),
    UnterminatedCString(MemoryPointer),
    DanglingPointerDeref,
    DoubleFree,
    InvalidMemoryAccess,
    InvalidFunctionPointer,
    InvalidBool,
    InvalidDiscriminant,
    PointerOutOfBounds {
        ptr: MemoryPointer,
        access: bool,
        allocation_size: u64,
    },
    InvalidNullPointerUsage,
    ReadPointerAsBytes,
    ReadBytesAsPointer,
    InvalidPointerMath,
    ReadUndefBytes,
    DeadLocal,
    InvalidBoolOp(mir::BinOp),
    Unimplemented(String),
    DerefFunctionPointer,
    ExecuteMemory,
    ArrayIndexOutOfBounds(Span, u64, u64),
    Math(Span, ConstMathErr),
    Intrinsic(String),
    OverflowingMath,
    InvalidChar(u128),
    OutOfMemory {
        allocation_size: u64,
        memory_size: u64,
        memory_usage: u64,
    },
    ExecutionTimeLimitReached,
    StackFrameLimitReached,
    OutOfTls,
    TlsOutOfBounds,
    AbiViolation(String),
    AlignmentCheckFailed {
        required: u64,
        has: u64,
    },
    MemoryLockViolation {
        ptr: MemoryPointer,
        len: u64,
        access: AccessKind,
        lock: LockInfo,
    },
    InvalidMemoryLockRelease {
        ptr: MemoryPointer,
        len: u64,
    },
    DeallocatedLockedMemory {
        ptr: MemoryPointer,
        lock: LockInfo,
    },
    ValidationFailure(String),
    CalledClosureAsFunction,
    VtableForArgumentlessMethod,
    ModifiedConstantMemory,
    AssumptionNotHeld,
    InlineAsm,
    TypeNotPrimitive(Ty<'tcx>),
    ReallocatedWrongMemoryKind(Kind, Kind),
    DeallocatedWrongMemoryKind(Kind, Kind),
    ReallocateNonBasePtr,
    DeallocateNonBasePtr,
    IncorrectAllocationInformation,
    Layout(layout::LayoutError<'tcx>),
    HeapAllocZeroBytes,
    HeapAllocNonPowerOfTwoAlignment(u64),
    Unreachable,
    Panic,
    NeedsRfc(String),
    NotConst(String),
    ReadFromReturnPointer,
    PathNotFound(Vec<String>),
}

pub type EvalResult<'tcx, T = ()> = Result<T, EvalError<'tcx>>;

impl<'tcx> Error for EvalError<'tcx> {
    fn description(&self) -> &str {
        use EvalError::*;
        match *self {
            FunctionPointerTyMismatch(..) =>
                "tried to call a function through a function pointer of a different type",
            InvalidMemoryAccess =>
                "tried to access memory through an invalid pointer",
            DanglingPointerDeref =>
                "dangling pointer was dereferenced",
            DoubleFree =>
                "tried to deallocate dangling pointer",
            InvalidFunctionPointer =>
                "tried to use an integer pointer or a dangling pointer as a function pointer",
            InvalidBool =>
                "invalid boolean value read",
            InvalidDiscriminant =>
                "invalid enum discriminant value read",
            PointerOutOfBounds { .. } =>
                "pointer offset outside bounds of allocation",
            InvalidNullPointerUsage =>
                "invalid use of NULL pointer",
            MemoryLockViolation { .. } =>
                "memory access conflicts with lock",
            ValidationFailure(..) =>
                "type validation failed",
            InvalidMemoryLockRelease { .. } =>
                "memory lock released that was never acquired",
            DeallocatedLockedMemory { .. } =>
                "tried to deallocate memory in conflict with a lock",
            ReadPointerAsBytes =>
                "a raw memory access tried to access part of a pointer value as raw bytes",
            ReadBytesAsPointer =>
                "a memory access tried to interpret some bytes as a pointer",
            InvalidPointerMath =>
                "attempted to do invalid arithmetic on pointers that would leak base addresses, e.g. comparing pointers into different allocations",
            ReadUndefBytes =>
                "attempted to read undefined bytes",
            DeadLocal =>
                "tried to access a dead local variable",
            InvalidBoolOp(_) =>
                "invalid boolean operation",
            Unimplemented(ref msg) => msg,
            DerefFunctionPointer =>
                "tried to dereference a function pointer",
            ExecuteMemory =>
                "tried to treat a memory pointer as a function pointer",
            ArrayIndexOutOfBounds(..) =>
                "array index out of bounds",
            Math(..) =>
                "mathematical operation failed",
            Intrinsic(..) =>
                "intrinsic failed",
            OverflowingMath =>
                "attempted to do overflowing math",
            NoMirFor(..) =>
                "mir not found",
            InvalidChar(..) =>
                "tried to interpret an invalid 32-bit value as a char",
            OutOfMemory{..} =>
                "could not allocate more memory",
            ExecutionTimeLimitReached =>
                "reached the configured maximum execution time",
            StackFrameLimitReached =>
                "reached the configured maximum number of stack frames",
            OutOfTls =>
                "reached the maximum number of representable TLS keys",
            TlsOutOfBounds =>
                "accessed an invalid (unallocated) TLS key",
            AbiViolation(ref msg) => msg,
            AlignmentCheckFailed{..} =>
                "tried to execute a misaligned read or write",
            CalledClosureAsFunction =>
                "tried to call a closure through a function pointer",
            VtableForArgumentlessMethod =>
                "tried to call a vtable function without arguments",
            ModifiedConstantMemory =>
                "tried to modify constant memory",
            AssumptionNotHeld =>
                "`assume` argument was false",
            InlineAsm =>
                "miri does not support inline assembly",
            TypeNotPrimitive(_) =>
                "expected primitive type, got nonprimitive",
            ReallocatedWrongMemoryKind(_, _) =>
                "tried to reallocate memory from one kind to another",
            DeallocatedWrongMemoryKind(_, _) =>
                "tried to deallocate memory of the wrong kind",
            ReallocateNonBasePtr =>
                "tried to reallocate with a pointer not to the beginning of an existing object",
            DeallocateNonBasePtr =>
                "tried to deallocate with a pointer not to the beginning of an existing object",
            IncorrectAllocationInformation =>
                "tried to deallocate or reallocate using incorrect alignment or size",
            Layout(_) =>
                "rustc layout computation failed",
            UnterminatedCString(_) =>
                "attempted to get length of a null terminated string, but no null found before end of allocation",
            HeapAllocZeroBytes =>
                "tried to re-, de- or allocate zero bytes on the heap",
            HeapAllocNonPowerOfTwoAlignment(_) =>
                "tried to re-, de-, or allocate heap memory with alignment that is not a power of two",
            Unreachable =>
                "entered unreachable code",
            Panic =>
                "the evaluated program panicked",
            NeedsRfc(_) =>
                "this feature needs an rfc before being allowed inside constants",
            NotConst(_) =>
                "this feature is not compatible with constant evaluation",
            ReadFromReturnPointer =>
                "tried to read from the return pointer",
            EvalError::PathNotFound(_) =>
                "a path could not be resolved, maybe the crate is not loaded",
        }
    }

    fn cause(&self) -> Option<&Error> { None }
}

impl<'tcx> fmt::Display for EvalError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use EvalError::*;
        match *self {
            PointerOutOfBounds { ptr, access, allocation_size } => {
                write!(f, "{} at offset {}, outside bounds of allocation {} which has size {}",
                       if access { "memory access" } else { "pointer computed" },
                       ptr.offset, ptr.alloc_id, allocation_size)
            },
            MemoryLockViolation { ptr, len, access, ref lock } => {
                write!(f, "{:?} access at {:?}, size {}, is in conflict with lock {:?}",
                       access, ptr, len, lock)
            }
            InvalidMemoryLockRelease { ptr, len } => {
                write!(f, "tried to release memory write lock at {:?}, size {}, but the write lock is held by someone else",
                       ptr, len)
            }
            DeallocatedLockedMemory { ptr, ref lock } => {
                write!(f, "tried to deallocate memory at {:?} in conflict with lock {:?}",
                       ptr, lock)
            }
            ValidationFailure(ref err) => {
                write!(f, "type validation failed: {}", err)
            }
            NoMirFor(ref func) => write!(f, "no mir for `{}`", func),
            FunctionPointerTyMismatch(sig, got) =>
                write!(f, "tried to call a function with sig {} through a function pointer of type {}", sig, got),
            ArrayIndexOutOfBounds(span, len, index) =>
                write!(f, "index out of bounds: the len is {} but the index is {} at {:?}", len, index, span),
            ReallocatedWrongMemoryKind(old, new) =>
                write!(f, "tried to reallocate memory from {:?} to {:?}", old, new),
            DeallocatedWrongMemoryKind(old, new) =>
                write!(f, "tried to deallocate {:?} memory but gave {:?} as the kind", old, new),
            Math(span, ref err) =>
                write!(f, "{:?} at {:?}", err, span),
            Intrinsic(ref err) =>
                write!(f, "{}", err),
            InvalidChar(c) =>
                write!(f, "tried to interpret an invalid 32-bit value as a char: {}", c),
            OutOfMemory { allocation_size, memory_size, memory_usage } =>
                write!(f, "tried to allocate {} more bytes, but only {} bytes are free of the {} byte memory",
                       allocation_size, memory_size - memory_usage, memory_size),
            AlignmentCheckFailed { required, has } =>
               write!(f, "tried to access memory with alignment {}, but alignment {} is required",
                      has, required),
            TypeNotPrimitive(ty) =>
                write!(f, "expected primitive type, got {}", ty),
            Layout(ref err) =>
                write!(f, "rustc layout computation failed: {:?}", err),
            NeedsRfc(ref msg) =>
                write!(f, "\"{}\" needs an rfc before being allowed inside constants", msg),
            NotConst(ref msg) =>
                write!(f, "Cannot evaluate within constants: \"{}\"", msg),
            EvalError::PathNotFound(ref path) =>
                write!(f, "Cannot find path {:?}", path),
            _ => write!(f, "{}", self.description()),
        }
    }
}
