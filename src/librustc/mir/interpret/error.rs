use std::error::Error;
use std::{fmt, env};
use std::rc::Rc;

use mir;
use ty::{FnSig, Ty, layout};

use super::{
    MemoryPointer, Lock, AccessKind
};

use rustc_const_math::ConstMathErr;
use syntax::codemap::Span;
use backtrace::Backtrace;

#[derive(Debug, Clone)]
pub struct EvalError<'tcx> {
    pub kind: Rc<EvalErrorKind<'tcx>>,
    pub backtrace: Option<Backtrace>,
}

impl<'tcx> From<EvalErrorKind<'tcx>> for EvalError<'tcx> {
    fn from(kind: EvalErrorKind<'tcx>) -> Self {
        let backtrace = match env::var("RUST_BACKTRACE") {
            Ok(ref val) if !val.is_empty() => Some(Backtrace::new_unresolved()),
            _ => None
        };
        EvalError {
            kind: Rc::new(kind),
            backtrace,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EvalErrorKind<'tcx> {
    /// This variant is used by machines to signal their own errors that do not
    /// match an existing variant
    MachineError(String),
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
        frame: usize,
        access: AccessKind,
        lock: Lock,
    },
    MemoryAcquireConflict {
        ptr: MemoryPointer,
        len: u64,
        kind: AccessKind,
        lock: Lock,
    },
    InvalidMemoryLockRelease {
        ptr: MemoryPointer,
        len: u64,
        frame: usize,
        lock: Lock,
    },
    DeallocatedLockedMemory {
        ptr: MemoryPointer,
        lock: Lock,
    },
    ValidationFailure(String),
    CalledClosureAsFunction,
    VtableForArgumentlessMethod,
    ModifiedConstantMemory,
    AssumptionNotHeld,
    InlineAsm,
    TypeNotPrimitive(Ty<'tcx>),
    ReallocatedWrongMemoryKind(String, String),
    DeallocatedWrongMemoryKind(String, String),
    ReallocateNonBasePtr,
    DeallocateNonBasePtr,
    IncorrectAllocationInformation(u64, usize, u64, u64),
    Layout(layout::LayoutError<'tcx>),
    HeapAllocZeroBytes,
    HeapAllocNonPowerOfTwoAlignment(u64),
    Unreachable,
    Panic,
    ReadFromReturnPointer,
    PathNotFound(Vec<String>),
    UnimplementedTraitSelection,
    /// Abort in case type errors are reached
    TypeckError,
}

pub type EvalResult<'tcx, T = ()> = Result<T, EvalError<'tcx>>;

impl<'tcx> Error for EvalError<'tcx> {
    fn description(&self) -> &str {
        use self::EvalErrorKind::*;
        match *self.kind {
            MachineError(ref inner) => inner,
            FunctionPointerTyMismatch(..) =>
                "tried to call a function through a function pointer of a different type",
            InvalidMemoryAccess =>
                "tried to access memory through an invalid pointer",
            DanglingPointerDeref =>
                "dangling pointer was dereferenced",
            DoubleFree =>
                "tried to deallocate dangling pointer",
            InvalidFunctionPointer =>
                "tried to use a function pointer after offsetting it",
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
            MemoryAcquireConflict { .. } =>
                "new memory lock conflicts with existing lock",
            ValidationFailure(..) =>
                "type validation failed",
            InvalidMemoryLockRelease { .. } =>
                "invalid attempt to release write lock",
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
            IncorrectAllocationInformation(..) =>
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
            ReadFromReturnPointer =>
                "tried to read from the return pointer",
            EvalErrorKind::PathNotFound(_) =>
                "a path could not be resolved, maybe the crate is not loaded",
            UnimplementedTraitSelection =>
                "there were unresolved type arguments during trait selection",
            TypeckError =>
                "encountered constants with type errors, stopping evaluation",
        }
    }
}

impl<'tcx> fmt::Display for EvalError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::EvalErrorKind::*;
        match *self.kind {
            PointerOutOfBounds { ptr, access, allocation_size } => {
                write!(f, "{} at offset {}, outside bounds of allocation {} which has size {}",
                       if access { "memory access" } else { "pointer computed" },
                       ptr.offset, ptr.alloc_id, allocation_size)
            },
            MemoryLockViolation { ptr, len, frame, access, ref lock } => {
                write!(f, "{:?} access by frame {} at {:?}, size {}, is in conflict with lock {:?}",
                       access, frame, ptr, len, lock)
            }
            MemoryAcquireConflict { ptr, len, kind, ref lock } => {
                write!(f, "new {:?} lock at {:?}, size {}, is in conflict with lock {:?}",
                       kind, ptr, len, lock)
            }
            InvalidMemoryLockRelease { ptr, len, frame, ref lock } => {
                write!(f, "frame {} tried to release memory write lock at {:?}, size {}, but cannot release lock {:?}",
                       frame, ptr, len, lock)
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
            ReallocatedWrongMemoryKind(ref old, ref new) =>
                write!(f, "tried to reallocate memory from {} to {}", old, new),
            DeallocatedWrongMemoryKind(ref old, ref new) =>
                write!(f, "tried to deallocate {} memory but gave {} as the kind", old, new),
            Math(_, ref err) =>
                write!(f, "{}", err.description()),
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
            PathNotFound(ref path) =>
                write!(f, "Cannot find path {:?}", path),
            MachineError(ref inner) =>
                write!(f, "machine error: {}", inner),
            IncorrectAllocationInformation(size, size2, align, align2) =>
                write!(f, "incorrect alloc info: expected size {} and align {}, got size {} and align {}", size, align, size2, align2),
            _ => write!(f, "{}", self.description()),
        }
    }
}
