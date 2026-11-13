//! regression test for <https://github.com/rust-lang/rust/issues/155030>

fn main() {
    let nums: [u32; 3] = [1, 2, 3];
    for num in nums {
        num *= 2; //~ ERROR cannot assign twice to immutable variable `num`
        println!("{num}");
    }
}
