//! Support code for dealing with libffi.

use libffi::low::CodePtr;
use libffi::middle::{Arg as ArgPtr, Cif, Type as FfiType};

/// Perform the actual FFI call.
///
/// # Safety
///
/// The safety invariants of the foreign function being called must be upheld (if any).
pub unsafe fn call<R: libffi::high::CType>(fun: CodePtr, args: &mut [OwnedArg]) -> R {
    let mut cif_args = vec![];
    let mut arg_ptrs = vec![];
    for a in args {
        cif_args.push(a.ty.take().unwrap());
        arg_ptrs.push(ArgPtr::new(&*a.bytes));
    }
    let cif = Cif::new(cif_args, R::reify().into_middle());
    // SAFETY: Caller upholds that the function is safe to call.
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
}
