// This test checks that the `_` type placeholder does not react
// badly if put as a lifetime parameter.

struct Foo<'a, T:'a> {
    r: &'a T
}

pub fn main() {
    let c: Foo<_, usize> = Foo { r: &5 };
    //~^ ERROR wrong number of type arguments: expected 1, found 2 [E0107]
}
