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
        //~^ ERROR: ==-comparison of unit values detected. This will always be true
        //~| NOTE: `-D clippy::unit-cmp` implied by `-D warnings`
        true;
    } == {
        false;
    } {}

    if {
        //~^ ERROR: >-comparison of unit values detected. This will always be false
        true;
    } > {
        false;
    } {}

    assert_eq!(
        //~^ ERROR: `assert_eq` of unit values detected. This will always succeed
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_eq!(
        //~^ ERROR: `debug_assert_eq` of unit values detected. This will always succeed
        {
            true;
        },
        {
            false;
        }
    );

    assert_ne!(
        //~^ ERROR: `assert_ne` of unit values detected. This will always fail
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_ne!(
        //~^ ERROR: `debug_assert_ne` of unit values detected. This will always fail
        {
            true;
        },
        {
            false;
        }
    );
}
