#![feature(dropck_eyepatch)]

// The point of this test is to illustrate that the `#[may_dangle]`
// attribute specifically allows, in the context of a type
// implementing `Drop`, a generic parameter to be instantiated with a
// lifetime that does not strictly outlive the owning type itself,
// and that this attributes effects are preserved when importing
// the type from another crate.
//
// See also dropck-eyepatch.rs for more information about the general
// structure of the test.

use std::cell::RefCell;

pub trait Foo { fn foo(&self, _: &str); }

pub struct Dt<A: Foo>(pub &'static str, pub A);
pub struct Dr<'a, B:'a+Foo>(pub &'static str, pub &'a B);
pub struct Pt<A,B: Foo>(pub &'static str, pub A, pub B);
pub struct Pr<'a, 'b, B:'a+'b+Foo>(pub &'static str, pub &'a B, pub &'b B);
pub struct St<A: Foo>(pub &'static str, pub A);
pub struct Sr<'a, B:'a+Foo>(pub &'static str, pub &'a B);

impl<A: Foo> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
impl<'a, B: Foo> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {}", self.0); self.1.foo(self.0); }
}
unsafe impl<#[may_dangle] A, B: Foo> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {}", self.0); self.2.foo(self.0); }
}
unsafe impl<#[may_dangle] 'a, 'b, B: Foo> Drop for Pr<'a, 'b, B> {
    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {}", self.0); self.2.foo(self.0); }
}

impl Foo for RefCell<String> {
    fn foo(&self, s: &str) {
        let s2 = format!("{}|{}", *self.borrow(), s);
        *self.borrow_mut() = s2;
    }
}

impl<'a, T:Foo> Foo for &'a T {
    fn foo(&self, s: &str) {
        (*self).foo(s);
    }
}
