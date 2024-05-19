//@ ignore-test (auxiliary, used by other tests)
#![crate_type = "lib"]

macro_rules! underscore {
    () => (
        _
    )
}
