// This test checks that the `_` type placeholder does not react
// badly if put as a lifetime parameter.

struct Foo<'a, T:'a> {
    r: &'a T
}

pub fn main() {
    let c: Foo<_, _> = Foo { r: &5 };
    //~^ ERROR this struct takes 1 type argument but 2 type arguments were supplied
}
