//@ known-bug: rust-lang/rust#128327

use std::ops::Deref;
struct Apple((Apple, <&'static [f64] as Deref>::Target(Banana ? Citron)));
fn main(){}
