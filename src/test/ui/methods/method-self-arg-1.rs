// Test method calls with self as an argument cannot subvert type checking.

struct Foo;

impl Foo {
    fn bar(&self) {}
}

fn main() {
    let x = Foo;
    Foo::bar(x); //~  ERROR mismatched types
                 //~| expected type `&Foo`
                 //~| found type `Foo`
                 //~| expected &Foo, found struct `Foo`
    Foo::bar(&42); //~  ERROR mismatched types
                      //~| expected type `&Foo`
                      //~| found type `&{integer}`
                      //~| expected struct `Foo`, found integer
}
