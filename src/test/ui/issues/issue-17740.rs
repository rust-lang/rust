struct Foo<'a> {
    data: &'a[u8],
}

impl <'a> Foo<'a>{
    fn bar(self: &mut Foo) {
    //~^ mismatched `self` parameter type
    //~| expected type `Foo<'a>`
    //~| found type `Foo<'_>`
    //~| lifetime mismatch
    //~| mismatched `self` parameter type
    //~| expected type `Foo<'a>`
    //~| found type `Foo<'_>`
    //~| lifetime mismatch
    }
}

fn main() {}
