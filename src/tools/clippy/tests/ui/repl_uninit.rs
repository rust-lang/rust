#![allow(deprecated, invalid_value, clippy::uninit_assumed_init)]
#![warn(clippy::mem_replace_with_uninit)]
//@no-rustfix
use std::mem;

fn might_panic<X>(x: X) -> X {
    // in practice this would be a possibly-panicky operation
    x
}

fn main() {
    let mut v = vec![0i32; 4];
    // the following is UB if `might_panic` panics
    unsafe {
        let taken_v = mem::replace(&mut v, mem::uninitialized());
        //~^ ERROR: replacing with `mem::uninitialized()`
        //~| NOTE: `-D clippy::mem-replace-with-uninit` implied by `-D warnings`
        let new_v = might_panic(taken_v);
        std::mem::forget(mem::replace(&mut v, new_v));
    }

    unsafe {
        let taken_v = mem::replace(&mut v, mem::MaybeUninit::uninit().assume_init());
        //~^ ERROR: replacing with `mem::MaybeUninit::uninit().assume_init()`
        let new_v = might_panic(taken_v);
        std::mem::forget(mem::replace(&mut v, new_v));
    }

    unsafe {
        let taken_v = mem::replace(&mut v, mem::zeroed());
        //~^ ERROR: replacing with `mem::zeroed()`
        let new_v = might_panic(taken_v);
        std::mem::forget(mem::replace(&mut v, new_v));
    }

    // this is silly but OK, because usize is a primitive type
    let mut u: usize = 42;
    let uref = &mut u;
    let taken_u = unsafe { mem::replace(uref, mem::zeroed()) };
    *uref = taken_u + 1;

    // this is still not OK, because uninit
    let taken_u = unsafe { mem::replace(uref, mem::uninitialized()) };
    //~^ ERROR: replacing with `mem::uninitialized()`
    *uref = taken_u + 1;
}
