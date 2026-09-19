//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// This function *looks* like it contains a call, but that call will be optimized out by MIR
// optimizations.
pub fn leaf_fn() -> String {
    String::new()
}

// This function contains a call, even after MIR optimizations. It is only eligible for
// cross-crate-inlining with "always".
pub fn stem_fn() -> String {
    inner()
}

#[inline(never)]
fn inner() -> String {
    String::from("test")
}

// This function's optimized MIR contains a call, but it is to an intrinsic.
pub fn leaf_with_intrinsic(a: &[u64; 2], b: &[u64; 2]) -> bool {
    a == b
}

// This function's optimized MIR contains assert terminators, not calls.
pub fn leaf_with_assert(a: i32, b: i32) -> i32 {
    a / b
}

pub trait FindToken<T> {
    fn find_token(&self, token: T) -> bool;
}

// This function's optimized MIR contains only statically dispatched trait calls.
// Their selected implementations are marked inline.
impl FindToken<char> for &str {
    fn find_token(&self, token: char) -> bool {
        self.chars().any(|character| character == token)
    }
}

pub fn free_find_token(input: &str, token: char) -> bool {
    input.chars().any(|character| character == token)
}

// This function's optimized MIR contains multiple statically dispatched trait calls.
// Their selected implementations are marked inline.
pub fn multiple_inline_trait_calls(input: &str, first: char, second: char) -> bool {
    input.chars().any(|character| character == first)
        && input.chars().any(|character| character == second)
}

pub struct InherentFindToken;

impl InherentFindToken {
    pub fn find_token(input: &str, token: char) -> bool {
        // Keep this distinct from `free_find_token` so LLVM cannot alias their definitions.
        !input.chars().any(|character| character == token)
    }
}

pub struct NonInlineTraitCall;

pub trait NonInlineTrait {
    fn call(&self, value: u32) -> u32;
}

impl NonInlineTrait for NonInlineTraitCall {
    #[inline(never)]
    fn call(&self, value: u32) -> u32 {
        value + 1
    }
}

pub fn stem_with_non_inline_trait_call(value: u32) -> u32 {
    <NonInlineTraitCall as NonInlineTrait>::call(&NonInlineTraitCall, value)
}

pub struct ExportedInlineTraitCall;

pub trait ExportedInlineTrait {
    fn exported_call(&self, value: u32) -> u32;
}

impl ExportedInlineTrait for ExportedInlineTraitCall {
    #[expect(unused_attributes)]
    #[inline]
    #[no_mangle]
    fn exported_call(&self, value: u32) -> u32 {
        value + 2
    }
}

pub fn stem_with_exported_inline_trait_call(value: u32) -> u32 {
    <ExportedInlineTraitCall as ExportedInlineTrait>::exported_call(&ExportedInlineTraitCall, value)
}

pub trait Factory<T> {
    type Item;
}

pub struct IntFactory;

impl<T> Factory<T> for IntFactory {
    type Item = usize;
}

pub trait InlineProjectionCall<T> {
    fn inline_call(value: T) -> T;
}

impl<T> InlineProjectionCall<T> for () {
    #[inline]
    fn inline_call(value: T) -> T {
        value
    }
}

// This is a compile-only regression test.
// The trivial bound makes normalization of the projected call argument fail
// while `cross_crate_inlinable` examines this generic function.
// The query must handle that failure without causing an ICE.
pub fn caller_with_trivial_bound<T>(
    value: <IntFactory as Factory<T>>::Item,
) -> <IntFactory as Factory<T>>::Item
where
    IntFactory: Factory<T>,
{
    <() as InlineProjectionCall<<IntFactory as Factory<T>>::Item>>::inline_call(value)
}
