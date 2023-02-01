trait Foo {
    fn foo(&self, x: u8) -> bool;
    fn bar(&self, x: u8, y: u8, z: u8);
    fn less(&self);
}

struct Bar;

impl Foo for Bar {
    fn foo(&self) -> bool { true } //~ ERROR E0050
    //~| HELP: modify the signature to match the trait definition
    fn bar(&self) { } //~ ERROR E0050
    //~| HELP: modify the signature to match the trait definition
    fn less(&self, x: u8, y: u8, z: u8) { } //~ ERROR E0050
    //~| HELP: modify the signature to match the trait definition
}

fn main() {
}
