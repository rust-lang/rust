// aux-build:cross_crate_ice.rs
// compile-pass

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
