//@ dont-require-annotations: NOTE

struct Foo<'a> {
    data: &'a[u8],
}

impl <'a> Foo<'a>{
    fn bar(self: &mut Foo) {
    //~^ ERROR mismatched `self` parameter type
    //~| NOTE expected struct `Foo<'a>`
    //~| NOTE found struct `Foo<'_>`
    //~| NOTE lifetime mismatch
    //~| ERROR mismatched `self` parameter type
    //~| NOTE expected struct `Foo<'a>`
    //~| NOTE found struct `Foo<'_>`
    //~| NOTE lifetime mismatch
    }
}

fn main() {}
