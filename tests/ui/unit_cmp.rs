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
        true;
    } == {
        false;
    } {}

    if {
        true;
    } > {
        false;
    } {}

    assert_eq!(
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_eq!(
        {
            true;
        },
        {
            false;
        }
    );

    assert_ne!(
        {
            true;
        },
        {
            false;
        }
    );
    debug_assert_ne!(
        {
            true;
        },
        {
            false;
        }
    );
}
