#![feature(optin_builtin_traits)]

trait Trait {
    type Bar;
}

struct Foo;

impl !Trait for Foo { } //~ ERROR E0192

fn main() {
}
