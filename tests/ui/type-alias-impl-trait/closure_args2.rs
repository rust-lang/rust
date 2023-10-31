// check-pass

#![feature(type_alias_impl_trait)]

trait Foo {
    // This was reachable in https://github.com/rust-lang/rust/issues/100800
    fn foo(&self) {
        unreachable!()
    }
}
impl<T> Foo for T {}

struct B;
impl B {
    fn foo(&self) {}
}

type Input = impl Foo;
fn run1<F: FnOnce(Input)>(f: F, i: Input) {
    f(i)
}
fn run2<F: FnOnce(B)>(f: F, i: B) {
    f(i)
}

fn bop() -> Input {
    run1(|x: B| x.foo(), B);
    run2(|x: B| x.foo(), B);
    panic!()
}

fn main() {}
