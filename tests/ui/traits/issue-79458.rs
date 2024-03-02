// Negative implementations should not be shown in trait suggestions.
// This is a regression test of #79458.

#[derive(Clone)]
struct Foo<'a, T> {
    bar: &'a mut T
    //~^ ERROR trait `Clone` is not implemented for `&mut T`
}

fn main() {}
