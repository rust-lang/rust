//@ revisions:rpass1 rpass2

// This test case makes sure re-order the methods in a vtable will
// trigger recompilation of codegen units that instantiate it.
//
// See https://github.com/rust-lang/rust/issues/89598

trait Foo {
    #[cfg(rpass1)]
    fn method1(&self) -> u32;

    fn method2(&self) -> u32;

    #[cfg(rpass2)]
    fn method1(&self) -> u32;
}

impl Foo for u32 {
    fn method1(&self) -> u32 { 17 }
    fn method2(&self) -> u32 { 42 }
}

fn main() {
    // Before #89598 was fixed, the vtable allocation would be cached during
    // a MIR optimization pass and then the codegen pass for the main object
    // file would not register a dependency on it (because of the missing
    // dep-tracking).
    //
    // In the rpass2 session, the main object file would not be re-compiled,
    // thus the mod1::foo(x) call would pass in an outdated vtable, while the
    // mod1 object would expect the new, re-ordered vtable, resulting in a
    // call to the wrong method.
    let x: &dyn Foo = &0u32;
    assert_eq!(mod1::foo(x), 17);
}

mod mod1 {
    pub(super) fn foo(x: &dyn super::Foo) -> u32 {
        x.method1()
    }
}
