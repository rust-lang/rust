// regression test of https://github.com/rust-lang/rust/pull/133572#issuecomment-2543007164
// see also the test for UniqueArc in variance-uniquearc.rs
//
// inline comments explain how this code *would* compile if UniqueRc was still covariant

#![feature(unique_rc_arc)]

use std::rc::UniqueRc;

fn extend_lifetime<'a, 'b>(x: &'a str) -> &'b str {
    let r = UniqueRc::new(""); // UniqueRc<&'static str>
    let w = UniqueRc::downgrade(&r); // Weak<&'static str>
    let mut r = r; // [IF COVARIANT]: ==>> UniqueRc<&'a str>
    *r = x; // assign the &'a str
    let _r = UniqueRc::into_rc(r); // Rc<&'a str>, but we only care to activate the weak
    let r = w.upgrade().unwrap(); // Rc<&'static str>
    *r // &'static str, coerces to &'b str
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let s = String::from("Hello World!");
    let r = extend_lifetime(&s);
    println!("{r}");
    drop(s);
    println!("{r}");
}
