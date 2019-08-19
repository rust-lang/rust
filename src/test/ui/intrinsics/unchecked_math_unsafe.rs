#![feature(core_intrinsics)]

fn main() {
    let (x, y) = (1u32, 2u32);
    let add = std::intrinsics::unchecked_add(x, y); //~ ERROR call to unsafe function
    let sub = std::intrinsics::unchecked_sub(x, y); //~ ERROR call to unsafe function
    let mul = std::intrinsics::unchecked_mul(x, y); //~ ERROR call to unsafe function
}
