// FIXME(#61117): Remove revisions once x86_64-gnu-debug CI job sets rust.debuginfo-level-tests=2
//@ revisions: no-debuginfo full-debuginfo
//@[no-debuginfo] compile-flags: -Cdebuginfo=0
//@[full-debuginfo] compile-flags: -Cdebuginfo=2
// the span for this revision comes from the LLVM debuginfo, which the gcc backend does not build
//@[full-debuginfo] ignore-backends: gcc
//@ build-fail
//@ compile-flags: --crate-type lib
//@ only-64bit Layout computation rejects this layout for different reasons on 32-bit.

#![feature(core_intrinsics)]
#![allow(internal_features)]

#[repr(C)]
pub struct Example([u8; isize::MAX as usize], [u16]);

// The `size_of_val` intrinsic is the first thing to ask for the layout of `Example` when there is
// no debuginfo, while with full debuginfo the type is described for the signature of `check`
// first. Both point at the code that requires the layout.
pub fn check(x: *const Example) -> usize {
    //[full-debuginfo]~^ ERROR are too big for the target architecture
    unsafe { std::intrinsics::size_of_val(x) }
    //[no-debuginfo]~^ ERROR are too big for the target architecture
}
