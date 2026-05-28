// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

fn main() {
    unsafe {
        let x = 12u8;
        let x_ptr: *const u8 = &x;
        let cast_ptr = x_ptr as *const Option<bool>;
        // Reading the discriminant should fail since the tag value is not in the valid
        // range for the tag field.
        let _val = matches!(*cast_ptr, None);
        //~^ ERROR: invalid tag
    }
}
