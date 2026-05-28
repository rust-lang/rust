// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation
#![feature(never_type)]

enum Never {}

// An enum with 4 variants of which only some are uninhabited -- so the uninhabited variants *do*
// have a discriminant.
#[allow(unused)]
enum UninhDiscriminant {
    A,
    B(!),
    C,
    D(Never),
}

fn main() {
    unsafe {
        let x = 3u8;
        let x_ptr: *const u8 = &x;
        let cast_ptr = x_ptr as *const UninhDiscriminant;
        // Reading the discriminant should fail since the tag value is not in the valid
        // range for the tag field.
        let _val = matches!(*cast_ptr, UninhDiscriminant::A);
        //~^ ERROR: invalid tag
    }
}
