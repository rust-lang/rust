//https://github.com/rust-lang/rust/issues/31364

const fn a() -> usize {
    //~^ ERROR cycle detected when const-evaluating + checking `a` [E0391]
    b()
}
const fn b() -> usize {
    a()
}
const ARR: [i32; a()] = [5; 6];

fn main() {}
