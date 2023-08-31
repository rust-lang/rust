#![warn(clippy::maybe_misused_cfg)]

fn main() {
    #[cfg(features = "not-really-a-feature")]
    //~^ ERROR: feature may misspelled as features
    //~| NOTE: `-D clippy::maybe-misused-cfg` implied by `-D warnings`
    let _ = 1 + 2;

    #[cfg(all(feature = "right", features = "wrong"))]
    //~^ ERROR: feature may misspelled as features
    let _ = 1 + 2;

    #[cfg(all(features = "wrong1", any(feature = "right", features = "wrong2", feature, features)))]
    //~^ ERROR: feature may misspelled as features
    //~| ERROR: feature may misspelled as features
    let _ = 1 + 2;
}
