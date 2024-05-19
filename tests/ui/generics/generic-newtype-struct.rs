//@ run-pass
//@ pretty-expanded FIXME #23616

struct S<T>(#[allow(dead_code)] T);

pub fn main() {
    let _s = S(2);
}
