//! Check that `vtable_size` gets overridden by llvm backend even if there is no
//! `rustc_intrinsic_must_be_overridden` attribute on this usage.
#![feature(rustc_attrs)]
//@run-pass

#[rustc_intrinsic]
pub unsafe fn vtable_size(_ptr: *const ()) -> usize {
    panic!();
}

trait Trait {}
impl Trait for () {}

fn main() {
    let x: &dyn Trait = &();
    unsafe {
        let (_data, vtable): (*const (), *const ()) = core::mem::transmute(x);
        assert_eq!(vtable_size(vtable), 0);
    }
}
