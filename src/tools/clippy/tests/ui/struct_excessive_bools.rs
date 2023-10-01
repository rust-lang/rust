#![warn(clippy::struct_excessive_bools)]

macro_rules! foo {
    () => {
        struct MacroFoo {
            a: bool,
            b: bool,
            c: bool,
            d: bool,
        }
    };
}

foo!();

struct Foo {
    a: bool,
    b: bool,
    c: bool,
}

struct BadFoo {
    //~^ ERROR: more than 3 bools in a struct
    a: bool,
    b: bool,
    c: bool,
    d: bool,
}

#[repr(C)]
struct Bar {
    a: bool,
    b: bool,
    c: bool,
    d: bool,
}

fn main() {
    struct FooFoo {
        //~^ ERROR: more than 3 bools in a struct
        a: bool,
        b: bool,
        c: bool,
        d: bool,
    }
}
