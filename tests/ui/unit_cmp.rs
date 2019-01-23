#![warn(clippy::unit_cmp)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

#[derive(PartialEq)]
pub struct ContainsUnit(()); // should be fine

fn main() {
    // this is fine
    if true == false {}

    // this warns
    if {
        true;
    } == {
        false;
    } {}

    if {
        true;
    } > {
        false;
    } {}
}
