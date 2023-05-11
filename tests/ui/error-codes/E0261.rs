fn foo(x: &'a str) { } //~ ERROR E0261
                       //~| undeclared lifetime

struct Foo {
    x: &'a str, //~ ERROR E0261
                //~| undeclared lifetime
}

fn main() {}
