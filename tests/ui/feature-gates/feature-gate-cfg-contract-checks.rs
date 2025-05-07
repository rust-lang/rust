#![crate_type = "lib"]

pub fn contract_checks_are_enabled() -> bool {
    cfg!(contract_checks) //~ ERROR `cfg(contract_checks)` is experimental
}
