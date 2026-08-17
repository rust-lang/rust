//@ ignore-32bit
// This test gives a different error on 32-bit architectures.
//@ stderr-per-bitwidth

union Transmute<T: Copy, U: Copy> {
    t: T,
    u: U,
}
trait Bar {
    fn bar(&self) -> u32;
}
struct Foo {
    foo: u32,
    bar: bool,
}
impl Bar for Foo {
    fn bar(&self) -> u32 {
        self.foo
    }
}
#[derive(Copy, Clone)]
struct Fat<'a>(&'a Foo, &'static VTable);
struct VTable {
    size: Foo,
}
const FOO: &dyn Bar = &Foo {
    foo: 128,
    bar: false,
};
const G: Fat = unsafe { Transmute { t: FOO }.u };
//~^ ERROR dangling reference

fn main() {}
