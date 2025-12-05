// This test checks that the `_` type placeholder does not react
// badly if put as a lifetime parameter.

struct Foo<'a, T:'a> {
    r: &'a T
}

pub fn main() {
    let c: Foo<_, _> = Foo { r: &5 };
    //~^ ERROR struct takes 1 generic argument but 2 generic arguments were supplied
}
