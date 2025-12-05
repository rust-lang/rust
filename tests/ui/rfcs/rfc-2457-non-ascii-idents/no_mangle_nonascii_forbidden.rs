#[no_mangle]
pub fn řųśť() {}  //~ ERROR `#[no_mangle]` requires ASCII identifier

pub struct Foo;

impl Foo {
    #[no_mangle]
    pub fn řųśť() {}  //~ ERROR `#[no_mangle]` requires ASCII identifier
}

trait Bar {
    fn řųśť();
}

impl Bar for Foo {
    #[no_mangle]
    fn řųśť() {}  //~ ERROR `#[no_mangle]` requires ASCII identifier
}

fn main() {}
