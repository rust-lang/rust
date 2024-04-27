trait Foo {
    fn foo(&self, x: u8) -> bool;
    fn bar(&self, x: u8, y: u8, z: u8);
    fn less(&self);
}

struct Bar;

impl Foo for Bar {
    fn foo(&self) -> bool { true } //~ ERROR E0050
    fn bar(&self) { } //~ ERROR E0050
    fn less(&self, x: u8, y: u8, z: u8) { } //~ ERROR E0050
}

fn main() {
}
