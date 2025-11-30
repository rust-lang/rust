//@ dont-require-annotations: NOTE

struct Foo<'a,'b> {
    x: &'a isize,
    y: &'b isize,
}

impl<'a,'b> Foo<'a,'b> {
    fn bar(self:
           Foo<'b,'a>
    //~^ ERROR mismatched `self` parameter type
    //~| NOTE expected struct `Foo<'a, 'b>`
    //~| NOTE found struct `Foo<'b, 'a>`
    //~| NOTE lifetime mismatch
    //~| ERROR mismatched `self` parameter type
    //~| NOTE expected struct `Foo<'a, 'b>`
    //~| NOTE found struct `Foo<'b, 'a>`
    //~| NOTE lifetime mismatch
           ) {}
}

fn main() {}
