//@ ignore-auxiliary (used by `./main.rs`)
#![crate_type = "lib"]

macro_rules! underscore {
    () => (
        _
    )
}
