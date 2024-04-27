// skip-filecheck
//@ test-mir-pass: Inline
//@ compile-flags: --crate-type=lib -C panic=abort

trait Foo {
    fn bar(&self) -> i32;
}

impl<T> Foo for T {
    fn bar(&self) -> i32 {
        0
    }
}

// EMIT_MIR inline_generically_if_sized.call.Inline.diff
pub fn call<T>(s: &T) -> i32 {
    s.bar()
}
