// run-pass
// pretty-expanded FIXME #23616

struct S<T>(T);

pub fn main() {
    let _s = S(2);
}
