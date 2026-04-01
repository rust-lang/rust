//@ known-bug: rust-lang/rust#125014
//@ compile-flags: -Znext-solver=coherence
#![feature(specialization)]

trait Foo {}

impl Foo for <u16 as Assoc>::Output {}

impl Foo for u32 {}

trait Assoc {
    type Output;
}
impl Output for u32 {}
impl Assoc for <u16 as Assoc>::Output {
    default type Output = bool;
}
