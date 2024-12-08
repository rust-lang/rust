//@ check-pass

#![warn(elided_lifetimes_in_paths,
        //~^ NOTE the lint level is defined here
        reason = "explicit anonymous lifetimes aid reasoning about ownership")]
#![warn(
    nonstandard_style,
    //~^ NOTE the lint level is defined here
    reason = r#"people shouldn't have to change their usual style habits
to contribute to our project"#
)]
#![allow(unused, reason = "unused code has never killed anypony")]

use std::fmt;

pub struct CheaterDetectionMechanism {}

impl fmt::Debug for CheaterDetectionMechanism {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        //~^ WARN hidden lifetime parameters in types are deprecated
        //~| NOTE expected lifetime parameter
        //~| NOTE explicit anonymous lifetimes aid
        //~| HELP indicate the anonymous lifetime
        fmt.debug_struct("CheaterDetectionMechanism").finish()
    }
}

fn main() {
    let Social_exchange_psychology = CheaterDetectionMechanism {};
    //~^ WARN should have a snake case name
    //~| NOTE #[warn(non_snake_case)]` implied by `#[warn(nonstandard_style)]
    //~| NOTE people shouldn't have to change their usual style habits
    //~| HELP convert the identifier to snake case
}
