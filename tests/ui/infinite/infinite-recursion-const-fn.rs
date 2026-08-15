//https://github.com/rust-lang/rust/issues/31364

const fn a() -> usize {
    b()
}
const fn b() -> usize {
    a()
}
const ARR: [i32; a()] = [5; 6]; //~ ERROR reached the configured maximum number of stack frames

fn main() {}
