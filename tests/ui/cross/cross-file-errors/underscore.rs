// We want this file only so we can test cross-file error
// messages, but we don't want it in an external crate.
// ignore-test
#![crate_type = "lib"]

macro_rules! underscore {
    () => (
        _
    )
}
