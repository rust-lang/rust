//! Lifetime-branded references to things manipulated by the LLVM bindings.

use std::marker::PhantomData;
use std::ptr;

/// FIXME(Zalathar): Use the stdlib's `PhantomInvariantLifetime` when its API is settled.
type PhantomInvariantLifetime<'a> = PhantomData<fn(&'a ()) -> &'a ()>;

/// Corresponds to `LLVMContextRef`.
#[derive(Clone, Copy)]
#[repr(transparent)]
#[cfg_attr(bootstrap, rustc_pass_by_value)]
#[cfg_attr(not(bootstrap), lint::pass_by_value)]
pub(crate) struct Context<'ll> {
    llcx_raw: ptr::NonNull<ContextRaw>,
    _invariant_ll: PhantomInvariantLifetime<'ll>,
}

/// Corresponds to `LLVMModuleRef`.
#[derive(Clone, Copy)]
#[repr(transparent)]
#[cfg_attr(bootstrap, rustc_pass_by_value)]
#[cfg_attr(not(bootstrap), lint::pass_by_value)]
pub(crate) struct Module<'ll> {
    llmod_raw: ptr::NonNull<ModuleRaw>,
    _invariant_ll: PhantomInvariantLifetime<'ll>,
}

impl<'ll> Module<'ll> {
    pub(crate) fn llmod_raw(self) -> ptr::NonNull<ModuleRaw> {
        self.llmod_raw
    }
}

unsafe extern "C" {
    /// Pointee of the raw pointer that is wrapped by a lifetime-branded [`Context`].
    pub(crate) type ContextRaw;
    /// Pointee of the raw pointer that is wrapped by a lifetime-branded [`Module`].
    pub(crate) type ModuleRaw;
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

/// Converts an unbranded pointer-to-module to a lifetime-branded [`Module`].
///
/// # Safety
/// - The pointer must be a valid LLVM module within the specified context.
/// - The module must not be disposed during the lifetime of its context.
pub(crate) unsafe fn wrap_llmod<'ll>(
    _llcx: Context<'ll>, // The context constrains `'ll` but isn't actually used.
    llmod_raw: ptr::NonNull<ModuleRaw>,
) -> Module<'ll> {
    Module { llmod_raw, _invariant_ll: PhantomData }
}

/// Corresponds to `LLVMTypeRef`.
///
/// LLVM types are interned within their LLVM context ([`Context<'ll>`]), so they
/// can be compared for equality (and hashed) based on their pointer address.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[cfg_attr(bootstrap, rustc_pass_by_value)]
#[cfg_attr(not(bootstrap), lint::pass_by_value)]
pub(crate) struct Type<'ll> {
    llty_raw: ptr::NonNull<TypeRaw>,
    _invariant_ll: PhantomInvariantLifetime<'ll>,
}

unsafe extern "C" {
    /// Pointee type of the raw pointer wrapped by [`Type`].
    type TypeRaw;
}

/// Corresponds to `LLVMValueRef`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
#[cfg_attr(bootstrap, rustc_pass_by_value)]
#[cfg_attr(not(bootstrap), lint::pass_by_value)]
pub(crate) struct Value<'ll> {
    llval_raw: ptr::NonNull<ValueRaw>,
    _invariant_ll: PhantomInvariantLifetime<'ll>,
}

unsafe extern "C" {
    /// Pointee type of the raw pointer wrapped by [`Value`].
    type ValueRaw;
}

/// Corresponds to `LLVMAttributeRef`.
#[derive(Clone, Copy)]
#[repr(transparent)]
#[rustc_pass_by_value]
pub(crate) struct Attribute<'ll> {
    llattr_raw: ptr::NonNull<AttributeRaw>,
    _invariant_ll: PhantomInvariantLifetime<'ll>,
}

unsafe extern "C" {
    /// Pointee type of the raw pointer wrapped by [`Attribute`].
    type AttributeRaw;
}
