#![warn(clippy::unit_cmp)]
#![allow(
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::derive_partial_eq_without_eq,
    clippy::needless_if
)]

#[derive(PartialEq)]
pub struct ContainsUnit(()); // should be fine

fn main() {
    // this is fine
    if true == false {}

    // this warns
    if {
        //~^ unit_cmp

        true;
    } == {
        false;
    } {}

    if {
        //~^ unit_cmp

        true;
    } > {
        false;
    } {}

    assert_eq!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_eq!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );

    assert_ne!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_ne!(
        //~^ unit_cmp
        {
            true;
        },
        {
            false;
        }
    );
}
