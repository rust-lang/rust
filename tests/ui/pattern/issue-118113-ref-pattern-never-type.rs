// check-pass

pub struct Foo {
    bar: u8
}

#[allow(unused_variables)]
fn main() {
    let Foo { ref bar } = loop {};
}
