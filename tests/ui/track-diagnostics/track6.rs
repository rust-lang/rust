// compile-flags: -Z track-diagnostics
// error-pattern: created at



pub trait Foo {
    fn bar();
}

impl <T> Foo for T {
    default fn bar() {}
}

fn main() {}
