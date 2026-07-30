// Test if the `copy_drop_in_place` lint is working correctly.

#[deny(copy_drop_in_place)]

fn main() {
    let mut x = 0u8;
    let y = &mut x as *mut _;
    unsafe {
        std::ptr::drop_in_place(y);
        //~^ ERROR calls to `std::ptr::drop_in_place` with a pointer to a Copy type does nothing
    }
}
