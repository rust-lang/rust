//@ revisions: default unchk_pass chk_pass chk_fail_ensures chk_fail_requires
//
//@ [default] run-pass
//@ [unchk_pass] run-pass
//@ [chk_pass] run-pass
//@ [chk_fail_requires] run-crash
//@ [chk_fail_ensures] run-crash
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_requires] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_ensures] compile-flags: -Zcontract-checks=yes
#![feature(cfg_contract_checks, contracts_internals, core_intrinsics)]

fn main() {
    #[cfg(any(default, unchk_pass))] // default: disabled
    assert_eq!(core::intrinsics::contract_checks(), false);

    #[cfg(chk_pass)] // explicitly enabled
    assert_eq!(core::intrinsics::contract_checks(), true);

    // always pass
    core::intrinsics::contract_check_requires(|| true);

    // fail if enabled
    #[cfg(any(default, unchk_pass, chk_fail_requires))]
    core::intrinsics::contract_check_requires(|| false);

    let doubles_to_two = { let old = 2; move |ret: &u32 | ret + ret == old };
    // Always pass
    core::intrinsics::contract_check_ensures(doubles_to_two, 1);

    // Fail if enabled
    #[cfg(any(default, unchk_pass, chk_fail_ensures))]
    core::intrinsics::contract_check_ensures(doubles_to_two, 2);
}
