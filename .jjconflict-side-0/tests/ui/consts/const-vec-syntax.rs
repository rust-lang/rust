//@ run-pass

fn f(_: &[isize]) {}

pub fn main() {
    let v = [ 1, 2, 3 ];
    f(&v);
}
