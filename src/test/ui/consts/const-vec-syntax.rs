// run-pass
// pretty-expanded FIXME #23616

fn f(_: &[isize]) {}

pub fn main() {
    let v = [ 1, 2, 3 ];
    f(&v);
}
