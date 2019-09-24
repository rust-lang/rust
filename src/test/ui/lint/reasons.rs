// build-pass (FIXME(62277): could be check-pass?)

#![feature(lint_reasons)]

#![warn(elided_lifetimes_in_paths,
        //~^ NOTE lint level defined here
        reason = "explicit anonymous lifetimes aid reasoning about ownership")]
#![warn(
    nonstandard_style,
    //~^ NOTE lint level defined here
    reason = r#"people shouldn't have to change their usual style habits
to contribute to our project"#
)]
#![allow(unused, reason = "unused code has never killed anypony")]

use std::fmt;

pub struct CheaterDetectionMechanism {}

impl fmt::Debug for CheaterDetectionMechanism {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        //~^ WARN hidden lifetime parameters in types are deprecated
        //~| NOTE explicit anonymous lifetimes aid
        //~| HELP indicate the anonymous lifetime
        fmt.debug_struct("CheaterDetectionMechanism").finish()
    }
}

fn main() {
    let Social_exchange_psychology = CheaterDetectionMechanism {};
    //~^ WARN should have a snake case name such as
    //~| NOTE people shouldn't have to change their usual style habits
}
