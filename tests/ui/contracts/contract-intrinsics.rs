//@ run-pass
//@ revisions: yes no none
//@ [yes] compile-flags: -Zcontract-checks=yes
//@ [no] compile-flags: -Zcontract-checks=no
#![feature(cfg_contract_checks, rustc_contracts, core_intrinsics)]

fn main() {
    #[cfg(none)] // default: disabled
    assert_eq!(core::intrinsics::contract_checks(), false);

    #[cfg(yes)] // explicitly enabled
    assert_eq!(core::intrinsics::contract_checks(), true);

    #[cfg(no)] // explicitly disabled
    assert_eq!(core::intrinsics::contract_checks(), false);

    assert_eq!(core::intrinsics::contract_check_requires(|| true), true);
    assert_eq!(core::intrinsics::contract_check_requires(|| false), false);

    let doubles_to_two = { let old = 2; move |ret| ret + ret == old };
    assert_eq!(core::intrinsics::contract_check_ensures(&1, doubles_to_two), true);
    assert_eq!(core::intrinsics::contract_check_ensures(&2, doubles_to_two), false);
}
