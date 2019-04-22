// rustc-env:RUST_NEW_ERROR_FORMAT

trait Foo {
    fn foo(x: u16);
    fn bar(&mut self, bar: &mut Bar);
}

struct Bar;

impl Foo for Bar {
    fn foo(x: i16) { } //~ ERROR incompatible type
    fn bar(&mut self, bar: &Bar) { } //~ ERROR incompatible type
}

fn main() {
}
