#![crate_type = "lib"]
#![deny(warnings)]

#[doc(test(no_crate_inject))] //~ ERROR
//~^ WARN
pub fn foo() {}

pub mod bar {
    #![doc(test(no_crate_inject))] //~ ERROR
    //~^ WARN
}
