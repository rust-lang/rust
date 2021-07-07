// aux-build:cross_crate_ice.rs

extern crate cross_crate_ice;

struct Bar(cross_crate_ice::Foo);
//~^ type alias impl traits are not allowed as field types in structs

impl Bar {
    fn zero(&self) -> &cross_crate_ice::Foo {
        &self.0
    }
}

fn main() {
    let _ = cross_crate_ice::foo();
}
