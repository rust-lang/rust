struct Foo<'a> {
    data: &'a[u8],
}

impl <'a> Foo<'a>{
    fn bar(self: &mut Foo) {
    //~^ mismatched method receiver
    //~| expected type `Foo<'a>`
    //~| found type `Foo<'_>`
    //~| lifetime mismatch
    //~| mismatched method receiver
    //~| expected type `Foo<'a>`
    //~| found type `Foo<'_>`
    //~| lifetime mismatch
    }
}

fn main() {}
