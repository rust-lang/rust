// Test for issue #23969


trait Foo {
    type Ty;
    const BAR: u32;
}

impl Foo for () {
    type Ty = ();
    type Ty = usize; //~ ERROR duplicate definitions
    const BAR: u32 = 7;
    const BAR: u32 = 8; //~ ERROR duplicate definitions
}

fn main() {
    let _: <() as Foo>::Ty = ();
    let _: u32 = <() as Foo>::BAR;
}
