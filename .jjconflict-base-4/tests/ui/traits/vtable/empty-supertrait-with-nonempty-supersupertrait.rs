//@ revisions: run dump
//@[run] run-pass
//@[dump] check-fail
//! Regression test for #145752
//! Ensure that `OneTwo` contains a vptr for `TwoAgain`
#![allow(unused)]
#![cfg_attr(dump, feature(rustc_attrs))]

trait One {
    fn one(&self) {
        panic!("don't call this");
    }
}
impl One for () {}

trait Two {
    fn two(&self) {
        println!("good");
    }
}
impl Two for () {}

trait TwoAgain: Two {}
impl<T: Two> TwoAgain for T {}

trait OneTwo: One + TwoAgain {}
impl<T: One + Two> OneTwo for T {}

fn main() {
    (&()).two();
    (&() as &dyn OneTwo).two();
    (&() as &dyn OneTwo as &dyn Two).two();

    // these two used to panic because they called `one` due to #145752
    (&() as &dyn OneTwo as &dyn TwoAgain).two();
    (&() as &dyn OneTwo as &dyn TwoAgain as &dyn Two).two();
}

#[cfg_attr(dump, rustc_dump_vtable)]
type T = dyn OneTwo;
//[dump]~^ ERROR vtable entries: [
//[dump]~| ERROR            MetadataDropInPlace,
//[dump]~| ERROR            MetadataSize,
//[dump]~| ERROR            MetadataAlign,
//[dump]~| ERROR            Method(<dyn OneTwo as One>::one - shim(reify)),
//[dump]~| ERROR            Method(<dyn OneTwo as Two>::two - shim(reify)),
//[dump]~| ERROR            TraitVPtr(<dyn OneTwo as Two>),
//[dump]~| ERROR            TraitVPtr(<dyn OneTwo as TwoAgain>),
//[dump]~| ERROR        ]
