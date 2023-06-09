// run-pass
// pretty-expanded FIXME #23616

fn f<T,>(_: T,) {}

struct Foo<T,>(#[allow(unused_tuple_struct_fields)] T);

struct Bar;

impl Bar {
    fn f(_: isize,) {}
    fn g(self, _: isize,) {}
    fn h(self,) {}
}

enum Baz {
    Qux(#[allow(unused_tuple_struct_fields)] isize,),
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
