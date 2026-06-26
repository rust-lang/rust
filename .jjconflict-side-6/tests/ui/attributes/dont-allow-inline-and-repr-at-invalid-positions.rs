//! Regression test for <https://github.com/rust-lang/rust/issues/31769>
fn main() {
    #[inline] struct Foo;  //~ ERROR attribute cannot be used on
    #[repr(C)] fn foo() {} //~ ERROR attribute cannot be used on
}
