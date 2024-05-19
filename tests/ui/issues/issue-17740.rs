struct Foo<'a> {
    data: &'a[u8],
}

impl <'a> Foo<'a>{
    fn bar(self: &mut Foo) {
    //~^ mismatched `self` parameter type
    //~| expected struct `Foo<'a>`
    //~| found struct `Foo<'_>`
    //~| lifetime mismatch
    //~| mismatched `self` parameter type
    //~| expected struct `Foo<'a>`
    //~| found struct `Foo<'_>`
    //~| lifetime mismatch
    }
}

fn main() {}
