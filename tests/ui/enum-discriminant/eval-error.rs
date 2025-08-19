union Foo {
    a: str,
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    //~| ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>`
}

enum Bar {
    Boo = {
        let _: Option<Foo> = None; //~ ERROR `Foo` has an unknown layout
        0
    },
}

union Foo2 {}
//~^ ERROR unions cannot have zero fields

enum Bar2 {
    Boo = {
        let _: Option<Foo2> = None;
        0
    },
}

#[repr(u8, packed)]
//~^ ERROR attribute should be applied to a struct or union
enum Foo3 {
    A
}

enum Bar3 {
    Boo = {
        let _: Option<Foo3> = None;
        0
    },
}

fn main() {}
