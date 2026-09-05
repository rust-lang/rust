//@ compile-flags: -Copt-level=3 -Zinline-mir=no -Zcross-crate-inline-threshold=100

#![crate_type = "lib"]
#![crate_name = "trait_call_eligibility_aux"]

// MIR inlining is disabled only in this auxiliary crate
// so every call under classification remains in optimized MIR.
// The downstream test crate uses normal O3 MIR inlining.

pub struct ExplicitlyInline;

pub trait ExplicitInlineCall {
    fn call(value: u32) -> u32;
}

impl ExplicitInlineCall for ExplicitlyInline {
    #[inline]
    fn call(value: u32) -> u32 {
        value.wrapping_add(10)
    }
}

pub fn encloses_explicit_inline_trait_call(value: u32) -> u32 {
    <ExplicitlyInline as ExplicitInlineCall>::call(value)
}

pub struct NotInline;

pub trait NoInlineCall {
    fn call(value: u32) -> u32;
}

impl NoInlineCall for NotInline {
    fn call(value: u32) -> u32 {
        value.wrapping_add(20)
    }
}

pub fn encloses_no_inline_trait_call(value: u32) -> u32 {
    <NotInline as NoInlineCall>::call(value)
}

#[inline]
fn direct_inline_call(value: u32) -> u32 {
    value.wrapping_add(30)
}

pub fn encloses_direct_inline_call(value: u32) -> u32 {
    direct_inline_call(value)
}

pub trait InlineVirtualCall {
    #[inline]
    fn call(&self, value: u32) -> u32 {
        value.wrapping_add(40)
    }
}

// Resolution produces an `InstanceKind::Virtual`, which must not be treated
// like the statically selected item above even though the provided trait method
// is explicitly inline.
pub fn encloses_inline_virtual_call(callee: &dyn InlineVirtualCall, value: u32) -> u32 {
    <dyn InlineVirtualCall as InlineVirtualCall>::call(callee, value)
}

pub trait UnresolvedCall {
    fn call(&self) -> u32;
}

// This is a compile-only regression test. Normalization succeeds, but instance selection cannot
// choose an implementation for `T`; cross-crate classification must conservatively reject the call.
pub fn caller_with_unresolved_selection<T: UnresolvedCall>(value: &T) -> u32 {
    <T as UnresolvedCall>::call(value)
}
