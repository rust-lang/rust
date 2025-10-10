//! Support code for dealing with libffi.

use libffi::low::CodePtr;
use libffi::middle::{Arg as ArgPtr, Cif, Type as FfiType};

/// Perform the actual FFI call.
///
/// # Safety
///
/// The safety invariants of the foreign function being called must be upheld (if any).
pub unsafe fn call<R: libffi::high::CType>(fun: CodePtr, args: &mut [OwnedArg]) -> R {
    let arg_ptrs: Vec<_> = args.iter().map(|arg| arg.ptr()).collect();
    let cif = Cif::new(args.iter_mut().map(|arg| arg.ty.take().unwrap()), R::reify().into_middle());
    // SAFETY: Caller upholds that the function is safe to call, and since we
    // were passed a slice reference we know the `OwnedArg`s won't have been
    // dropped by this point.
    unsafe { cif.call(fun, &arg_ptrs) }
}

/// An argument for an FFI call.
#[derive(Debug, Clone)]
pub struct OwnedArg {
    /// The type descriptor for this argument.
    ty: Option<FfiType>,
    /// Corresponding bytes for the value.
    bytes: Box<[u8]>,
}

impl OwnedArg {
    /// Instantiates an argument from a type descriptor and bytes.
    pub fn new(ty: FfiType, bytes: Box<[u8]>) -> Self {
        Self { ty: Some(ty), bytes }
    }

    /// Creates a libffi argument pointer pointing to this argument's bytes.
    /// NB: Since `libffi::middle::Arg` ignores the lifetime of the reference
    /// it's derived from, it is up to the caller to ensure the `OwnedArg` is
    /// not dropped before unsafely calling `libffi::middle::Cif::call()`!
    fn ptr(&self) -> ArgPtr {
        // FIXME: Using `&self.bytes[0]` to reference the whole array is
        // definitely unsound under SB, but we're waiting on
        // https://github.com/libffi-rs/libffi-rs/commit/112a37b3b6ffb35bd75241fbcc580de40ba74a73
        // to land in a release so that we don't need to do this.
        ArgPtr::new(&self.bytes[0])
    }
}
