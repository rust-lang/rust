//@ revisions: default chk_fail_ensures chk_fail_requires
//
//@ [default] run-pass
//@ [chk_fail_requires] run-crash
//@ [chk_fail_ensures] run-crash
#![feature(contracts_internals, core_intrinsics)]

fn main() {
    // always pass
    core::intrinsics::contract_check_requires(|| true);

    // always fail
    #[cfg(chk_fail_requires)]
    core::intrinsics::contract_check_requires(|| false);

    let doubles_to_two = { let old = 2; move |ret: &u32 | ret + ret == old };
    // Always pass
    core::intrinsics::contract_check_ensures(Some(doubles_to_two), 1);

    // always fail
    #[cfg(chk_fail_ensures)]
    core::intrinsics::contract_check_ensures(Some(doubles_to_two), 2);
}
