// Test method calls with self as an argument cannot subvert type checking.

//@ dont-require-annotations: NOTE

struct Foo;

impl Foo {
    fn bar(&self) {}
}

fn main() {
    let x = Foo;
    Foo::bar(x); //~  ERROR mismatched types
                 //~| NOTE expected `&Foo`, found `Foo`
    Foo::bar(&42); //~  ERROR mismatched types
                      //~| NOTE expected `&Foo`, found `&{integer}`
                      //~| NOTE expected reference `&Foo`
                      //~| NOTE found reference `&{integer}`
}
