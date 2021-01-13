// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait T<const A: usize> {
    fn f();
}
struct S;

impl T<0usize> for S {
    fn f() {}
}

fn main() {
    let _err = <S as T<0usize>>::f();
}
