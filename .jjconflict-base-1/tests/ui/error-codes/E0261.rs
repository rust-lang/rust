fn foo(x: &'a str) { } //~ ERROR E0261
                       //~| NOTE undeclared lifetime

struct Foo {
    x: &'a str, //~ ERROR E0261
                //~| NOTE undeclared lifetime
}

fn main() {}
