//@ compile-flags:-O

#[inline]
fn foo() {}

pub static ADDR: fn() = foo;

#[inline(always)]
pub fn bar(x: fn()) -> bool {
    x == ADDR
}
