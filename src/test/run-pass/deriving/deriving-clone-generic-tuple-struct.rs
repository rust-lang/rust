// run-pass
// pretty-expanded FIXME #23616

#[derive(Clone)]
struct S<T>(T, ());

pub fn main() {
    let _ = S(1, ()).clone();
}
