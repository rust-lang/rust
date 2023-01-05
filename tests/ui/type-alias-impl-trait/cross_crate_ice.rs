// aux-build:cross_crate_ice.rs
// build-pass (FIXME(62277): could be check-pass?)

extern crate cross_crate_ice;

struct Bar(cross_crate_ice::Foo);

impl Bar {
    fn zero(&self) -> &cross_crate_ice::Foo {
        &self.0
    }
}

fn main() {
    let _ = cross_crate_ice::foo();
}
