// run-pass
// #24947 ICE using a trait-associated const in an array size


struct Foo;

impl Foo {
    const SIZE: usize = 8;
}

trait Bar {
    const BAR_SIZE: usize;
}

impl Bar for Foo {
    const BAR_SIZE: usize = 12;
}

#[allow(unused_variables)]
fn main() {
    let w: [u8; 12] = [0u8; <Foo as Bar>::BAR_SIZE];
    let x: [u8; 12] = [0u8; <Foo>::BAR_SIZE];
    let y: [u8; 8] = [0u8; <Foo>::SIZE];
    let z: [u8; 8] = [0u8; Foo::SIZE];
}
