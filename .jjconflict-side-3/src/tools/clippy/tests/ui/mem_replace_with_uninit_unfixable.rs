// The lint does not offer a suggestion for the `zeroed` case
//@ no-rustfix
#![warn(clippy::mem_replace_with_uninit)]
#![expect(invalid_value)]

use std::mem;

fn might_panic<X>(x: X) -> X {
    // in practice this would be a possibly-panicky operation
    x
}

fn main() {
    let mut v = vec![0i32; 4];
    unsafe {
        let taken_v = mem::replace(&mut v, mem::zeroed());
        //~^ mem_replace_with_uninit

        let new_v = might_panic(taken_v);
        std::mem::forget(mem::replace(&mut v, new_v));
    }
}
