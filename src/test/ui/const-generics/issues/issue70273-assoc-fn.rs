// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

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
