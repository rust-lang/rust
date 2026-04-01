//@ dont-require-annotations: NOTE
//! regression test for <https://github.com/rust-lang/rust/issues/17740>

struct Foo<'a, 'b> {
    x: &'a isize,
    y: &'b isize,
}

impl<'a, 'b> Foo<'a, 'b> {
    fn bar(
        self: Foo<'b, 'a>,
        //~^ ERROR mismatched `self` parameter type
        //~| NOTE expected struct `Foo<'a, 'b>`
        //~| NOTE found struct `Foo<'b, 'a>`
        //~| NOTE lifetime mismatch
        //~| ERROR mismatched `self` parameter type
        //~| NOTE expected struct `Foo<'a, 'b>`
        //~| NOTE found struct `Foo<'b, 'a>`
        //~| NOTE lifetime mismatch
    ) {
    }
}

struct Bar<'a> {
    data: &'a [u8],
}

impl<'a> Bar<'a> {
    fn bar(self: &mut Bar) {
        //~^ ERROR mismatched `self` parameter type
        //~| NOTE expected struct `Bar<'a>`
        //~| NOTE found struct `Bar<'_>`
        //~| NOTE lifetime mismatch
        //~| ERROR mismatched `self` parameter type
        //~| NOTE expected struct `Bar<'a>`
        //~| NOTE found struct `Bar<'_>`
        //~| NOTE lifetime mismatch
    }
}

fn main() {}
