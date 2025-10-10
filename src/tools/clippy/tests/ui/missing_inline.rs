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
//~^ missing_inline_in_public_items

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
    //~^ missing_inline_in_public_items

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
    //~^ missing_inline_in_public_items

    // missing #[inline]
    fn PubBar_b() {}
    //~^ missing_inline_in_public_items

    // missing #[inline]
    fn PubBar_c() {}
    //~^ missing_inline_in_public_items
}

// do not need inline because Foo is not exported
impl Foo {
    fn FooImpl() {} // ok
}

// need inline because PubFoo is exported
impl PubFoo {
    // missing #[inline]
    pub fn PubFooImpl() {}
    //~^ missing_inline_in_public_items
}

// do not lint this since users cannot control the external code
#[derive(Debug)]
pub struct S;

pub mod issue15301 {
    #[unsafe(no_mangle)]
    pub extern "C" fn call_from_c() {
        println!("Just called a Rust function from C!");
    }

    #[unsafe(no_mangle)]
    pub extern "Rust" fn call_from_rust() {
        println!("Just called a Rust function from Rust!");
    }

    #[unsafe(no_mangle)]
    pub fn call_from_rust_no_extern() {
        println!("Just called a Rust function from Rust!");
    }
}

pub mod issue15491 {
    pub trait Foo {
        #[allow(clippy::missing_inline_in_public_items)]
        fn foo(&self) {}
    }
}
