// run-pass
#![allow(dead_code, type_param_on_variant_ctor)]
// pretty-expanded FIXME #23616

#[derive(Clone)]
enum E<T,U> {
    A(T),
    B(T,U),
    C
}

pub fn main() {
    let _ = E::A::<isize, isize>(1).clone();
}
