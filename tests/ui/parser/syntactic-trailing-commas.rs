//@ run-pass

fn f<T,>(_: T,) {}

struct Foo<T,>(#[allow(dead_code)] T);

struct Bar;

impl Bar {
    fn f(_: isize,) {}
    fn g(self, _: isize,) {}
    fn h(self,) {}
}

enum Baz {
    Qux(#[allow(dead_code)] isize,),
}

#[allow(unused,)]
pub fn main() {
    f::<isize,>(0,);
    let (_, _,) = (1, 1,);
    let [_, _,] = [1, 1,];
    let [_, _, .., _,] = [1, 1, 1, 1,];
    let [_, _, _, ..,] = [1, 1, 1, 1,];

    let x: Foo<isize,> = Foo::<isize,>(1);

    Bar::f(0,);
    Bar.g(0,);
    Bar.h();

    let x = Baz::Qux(1,);
}
