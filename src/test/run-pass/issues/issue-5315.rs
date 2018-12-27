// run-pass
// pretty-expanded FIXME #23616

struct A(bool);

pub fn main() {
    let f = A;
    f(true);
}
