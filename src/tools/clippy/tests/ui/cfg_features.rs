#![warn(clippy::maybe_misused_cfg)]

fn main() {
    #[cfg(features = "not-really-a-feature")]
    //~^ ERROR: 'feature' may be misspelled as 'features'
    //~| NOTE: `-D clippy::maybe-misused-cfg` implied by `-D warnings`
    let _ = 1 + 2;

    #[cfg(all(feature = "right", features = "wrong"))]
    //~^ ERROR: 'feature' may be misspelled as 'features'
    let _ = 1 + 2;

    #[cfg(all(features = "wrong1", any(feature = "right", features = "wrong2", feature, features)))]
    //~^ ERROR: 'feature' may be misspelled as 'features'
    //~| ERROR: 'feature' may be misspelled as 'features'
    let _ = 1 + 2;

    #[cfg(tests)]
    //~^ ERROR: 'test' may be misspelled as 'tests'
    let _ = 2;
    #[cfg(Test)]
    //~^ ERROR: 'test' may be misspelled as 'Test'
    let _ = 2;

    #[cfg(all(tests, Test))]
    //~^ ERROR: 'test' may be misspelled as 'tests'
    //~| ERROR: 'test' may be misspelled as 'Test'
    let _ = 2;
}
