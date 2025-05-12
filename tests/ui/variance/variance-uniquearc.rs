// regression test of https://github.com/rust-lang/rust/pull/133572#issuecomment-2543007164
// see also the test for UniqueRc` in variance-uniquerc.rs
//
// inline comments explain how this code *would* compile if UniqueArc was still covariant

#![feature(unique_rc_arc)]

use std::sync::UniqueArc;

fn extend_lifetime<'a, 'b>(x: &'a str) -> &'b str {
    let r = UniqueArc::new(""); // UniqueArc<&'static str>
    let w = UniqueArc::downgrade(&r); // Weak<&'static str>
    let mut r = r; // [IF COVARIANT]: ==>> UniqueArc<&'a str>
    *r = x; // assign the &'a str
    let _r = UniqueArc::into_arc(r); // Arc<&'a str>, but we only care to activate the weak
    let r = w.upgrade().unwrap(); // Arc<&'static str>
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
