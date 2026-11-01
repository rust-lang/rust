#![crate_type = "lib"]

pub fn ub_checks_are_enabled() -> bool {
    cfg!(ub_checks) //~ ERROR `cfg(ub_checks)` is experimental
}
