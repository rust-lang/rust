//@ compile-flags: --crate-type=lib --emit=mir -C debuginfo=0
pub trait Trait {}
impl<T> Trait for T {}

//~? ERROR: values of the type `[u8; usize::MAX]` are too big for the target architecture
pub fn foo(x: &[u8; usize::MAX]) -> &dyn Trait {
    x as &dyn Trait
}
