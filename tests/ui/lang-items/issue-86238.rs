// Regression test for the ICE described in issue #86238.

#![feature(lang_items)]
#![feature(no_core)]

#![no_core]
fn main() {
    let one = || {};
    one()
    //~^ ERROR: failed to find an overloaded call trait for closure call
    //~| HELP: make sure the `fn`/`fn_mut`/`fn_once` lang items are defined
}
#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
