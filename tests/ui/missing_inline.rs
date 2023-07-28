#![warn(clippy::missing_inline_in_public_items)]
#![crate_type = "dylib"]
// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.
#![allow(dead_code, non_snake_case)]

type Typedef = String;
pub type PubTypedef = String;

struct Foo; // ok
pub struct PubFoo; // ok
enum FooE {} // ok
pub enum PubFooE {} // ok

mod module {} // ok
pub mod pub_module {} // ok

fn foo() {}
// missing #[inline]
pub fn pub_foo() {}
//~^ ERROR: missing `#[inline]` for a function
//~| NOTE: `-D clippy::missing-inline-in-public-items` implied by `-D warnings`
#[inline]
pub fn pub_foo_inline() {} // ok
#[inline(always)]
pub fn pub_foo_inline_always() {} // ok

#[allow(clippy::missing_inline_in_public_items)]
pub fn pub_foo_no_inline() {}

trait Bar {
    fn Bar_a(); // ok
    fn Bar_b() {} // ok
}

pub trait PubBar {
    fn PubBar_a(); // ok
    // missing #[inline]
    fn PubBar_b() {}
    //~^ ERROR: missing `#[inline]` for a default trait method
    #[inline]
    fn PubBar_c() {} // ok
}

// none of these need inline because Foo is not exported
impl PubBar for Foo {
    fn PubBar_a() {} // ok
    fn PubBar_b() {} // ok
    fn PubBar_c() {} // ok
}

// all of these need inline because PubFoo is exported
impl PubBar for PubFoo {
    // missing #[inline]
    fn PubBar_a() {}
    //~^ ERROR: missing `#[inline]` for a method
    // missing #[inline]
    fn PubBar_b() {}
    //~^ ERROR: missing `#[inline]` for a method
    // missing #[inline]
    fn PubBar_c() {}
    //~^ ERROR: missing `#[inline]` for a method
}

// do not need inline because Foo is not exported
impl Foo {
    fn FooImpl() {} // ok
}

// need inline because PubFoo is exported
impl PubFoo {
    // missing #[inline]
    pub fn PubFooImpl() {}
    //~^ ERROR: missing `#[inline]` for a method
}

// do not lint this since users cannot control the external code
#[derive(Debug)]
pub struct S;
