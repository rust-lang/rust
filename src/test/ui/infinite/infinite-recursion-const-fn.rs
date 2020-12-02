//https://github.com/rust-lang/rust/issues/31364

const fn a() -> usize {
    b() //~ 4:8: evaluation of constant value failed [E0080]
}
const fn b() -> usize {
    a()
}
const ARR: [i32; a()] = [5; 6];

fn main() {}
