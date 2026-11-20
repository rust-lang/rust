//@ run-pass
//@ aux-build:const_mut_refs_crate.rs

//! Regression test for https://github.com/rust-lang/rust/issues/79738
//! Show how we are not duplicating allocations anymore. Statics that
//! copy their value from another static used to also duplicate
//! memory behind references.

extern crate const_mut_refs_crate as other;

use other::{
    inner::{INNER_MOD_BAR, INNER_MOD_FOO},
    BAR, FOO,
};

pub static LOCAL_FOO: &'static i32 = &41;
pub static LOCAL_BAR: &'static i32 = LOCAL_FOO;
pub static mut COPY_OF_REMOTE_FOO: &'static mut i32 = unsafe { FOO };

static DOUBLE_REF: &&i32 = &&99;
static ONE_STEP_ABOVE: &i32 = *DOUBLE_REF;
static mut DOUBLE_REF_MUT: &mut &mut i32 = &mut &mut 99;
static mut ONE_STEP_ABOVE_MUT: &mut i32 = unsafe { *DOUBLE_REF_MUT };

pub fn main() {
    unsafe {
        assert_eq!(FOO as *const i32, BAR as *const i32);
        assert_eq!(INNER_MOD_FOO as *const i32, INNER_MOD_BAR as *const i32);
        assert_eq!(LOCAL_FOO as *const i32, LOCAL_BAR as *const i32);
        assert_eq!(*DOUBLE_REF as *const i32, ONE_STEP_ABOVE as *const i32);
        assert_eq!(*DOUBLE_REF_MUT as *mut i32, ONE_STEP_ABOVE_MUT as *mut i32);

        assert_eq!(FOO as *const i32, COPY_OF_REMOTE_FOO as *const i32);
    }
}
