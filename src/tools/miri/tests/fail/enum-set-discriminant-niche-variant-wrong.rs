#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::num::NonZeroI32;

// We define our own option type so that we can control the varian indices.
#[allow(unused)]
enum Option<T> {
    None,
    Some(T),
}
use Option::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn set_discriminant(ptr: &mut Option<NonZeroI32>) {
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
