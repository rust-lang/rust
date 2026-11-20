// At the time of writing, vtable.rs would ICE only with debuginfo disabled, while this testcase,
// originally reported as #152030, would ICE even with debuginfo enabled.
//@ revisions: no-debuginfo full-debuginfo
//@ compile-flags: --crate-type=lib --emit=mir
//@[no-debuginfo] compile-flags: -C debuginfo=0
//@[full-debuginfo] compile-flags: -C debuginfo=2
#![feature(try_as_dyn)]

trait Trait {}
impl<T> Trait for T {}

//~? ERROR: values of the type `[u8; usize::MAX]` are too big for the target architecture
pub fn foo(x: &[u8; usize::MAX]) {
    let _ = std::any::try_as_dyn::<[u8; usize::MAX], dyn Trait>(x);
}
