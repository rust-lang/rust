#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::num::NonZero;

// We define our own option type so that we can control the variant indices.
#[allow(unused)]
enum Option<T> {
    None,    // variant 0
    Some(T), // variant 1
}
use Option::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn set_discriminant(ptr: &mut Option<NonZero<i32>>) {
    mir! {
        {
            // We set the discriminant to `Some`, which is a NOP since this is the niched variant.
            // However, the enum is actually encoding `None` currently! That's not good...
            SetDiscriminant(*ptr, 1);
            //~^ ERROR: trying to set discriminant of a Option<std::num::NonZero<i32>> to the niched variant, but the value does not match
            Return()
        }
    }
}

pub fn main() {
    let mut v = None;
    set_discriminant(&mut v);
}
