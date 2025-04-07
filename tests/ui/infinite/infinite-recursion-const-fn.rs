//https://github.com/rust-lang/rust/issues/31364

const fn a() -> usize {
    b()
}
const fn b() -> usize {
    a()
}
const ARR: [i32; a()] = [5; 6]; //~ ERROR evaluation of constant value failed [E0080]

fn main() {}
