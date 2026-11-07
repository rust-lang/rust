// gate-test-associated_type_defaults

trait Foo {
    type Bar = u8; //~ ERROR associated type defaults are unstable
}

fn main() {}
