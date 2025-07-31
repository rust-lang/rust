//@ compile-flags: -Clto=thin -C codegen-units=8

#[inline]
pub fn foo(b: u8) {
    b.to_string();
}
