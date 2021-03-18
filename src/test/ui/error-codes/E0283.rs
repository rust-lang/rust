trait Generator {
    fn create() -> u32;
}

struct Impl;

impl Generator for Impl {
    fn create() -> u32 { 1 }
}

impl Impl {
    fn new() -> Self {
        Impl{}
    }
}

impl Into<u32> for Impl {
    fn into(self) -> u32 { 1 }
}

fn foo(bar: u32) {}

struct AnotherImpl;

impl Generator for AnotherImpl {
    fn create() -> u32 { 2 }
}

fn main() {
    let cont: u32 = Generator::create(); //~ ERROR E0283
}

fn buzz() {
    let foo_impl = Impl::new();
    let bar = foo_impl.into() * 1u32; //~ ERROR E0283
    foo(bar);
}
