//@ run-pass
fn f(x: &isize) -> &isize {
    return &*x;
}

pub fn main() {
    let three = &3;
    println!("{}", *f(three));
}
