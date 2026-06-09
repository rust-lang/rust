fn main() {
    let (x, y) = (1u32, 2u32);
    unsafe {
        let add = std::intrinsics::unchecked_add(x, y); //~ ERROR use of unstable library feature
        let sub = std::intrinsics::unchecked_sub(x, y); //~ ERROR use of unstable library feature
        let mul = std::intrinsics::unchecked_mul(x, y); //~ ERROR use of unstable library feature
    }
}
