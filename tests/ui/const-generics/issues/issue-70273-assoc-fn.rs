//@ check-pass

trait T<const A: usize> {
    fn f();
}
struct S;

impl T<0usize> for S {
    fn f() {}
}

fn main() {
    <S as T<0usize>>::f();
}
