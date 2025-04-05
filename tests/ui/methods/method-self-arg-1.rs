// Test method calls with self as an argument cannot subvert type checking.

struct Foo;

impl Foo {
    fn bar(&self) {}
}

fn main() {
    let x = Foo;
    Foo::bar(x); //~  ERROR mismatched types
                 //~| NOTE_NONVIRAL expected `&Foo`, found `Foo`
    Foo::bar(&42); //~  ERROR mismatched types
                      //~| NOTE_NONVIRAL expected `&Foo`, found `&{integer}`
                      //~| NOTE_NONVIRAL expected reference `&Foo`
                      //~| NOTE_NONVIRAL found reference `&{integer}`
}
