//! Lifetime-branded references to things manipulated by the LLVM bindings.

use std::marker::PhantomData;
use std::ptr;

/// Corresponds to `LLVMContextRef`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct Context<'ll> {
    llcx_raw: ptr::NonNull<ContextRaw>,
    _invariant_ll: PhantomData<fn(&'ll ()) -> &'ll ()>,
}

unsafe extern "C" {
    /// Pointee of the raw pointer that is wrapped by a lifetime-branded [`Context`].
    pub(crate) type ContextRaw;
}

/// Converts an unbranded pointer-to-context to a lifetime-branded [`Context`].
///
/// # Safety
/// - The pointer must be a valid LLVM context.
/// - If the context is passed to safe code, its caller-requested lifetime must be distinct
///   from that of any other context, and must not outlive the underlying context.
pub(crate) unsafe fn wrap_llcx<'ll>(llcx_raw: ptr::NonNull<ContextRaw>) -> Context<'ll> {
    Context { llcx_raw, _invariant_ll: PhantomData }
}
