#![feature(core_intrinsics)]
fn main() {
    // MIN/-1 cannot be represented
    unsafe {
        std::intrinsics::unchecked_div(i16::MIN, -1); //~ ERROR: overflow in signed division (dividing MIN by -1)
    }
}
