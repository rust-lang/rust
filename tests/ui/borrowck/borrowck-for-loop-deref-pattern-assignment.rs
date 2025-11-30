fn main() {
    let nums: &[u32] = &[1, 2, 3];
    for &num in nums {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`
        println!("{num}");
    }
}
