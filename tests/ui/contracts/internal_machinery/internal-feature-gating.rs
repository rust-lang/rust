// gate-test-rustc_contracts_internals

fn main() {
    // intrinsics are guarded by rustc_contracts_internals feature gate.
    core::intrinsics::contract_checks();
    //~^ ERROR use of unstable library feature `rustc_contracts_internals`
    core::intrinsics::contract_check_requires(|| true);
    //~^ ERROR use of unstable library feature `rustc_contracts_internals`
    core::intrinsics::contract_check_ensures(&1, |_|true);
    //~^ ERROR use of unstable library feature `rustc_contracts_internals`

    // lang items are guarded by rustc_contracts_internals feature gate.
    core::contracts::check_requires(|| true);
    //~^ ERROR use of unstable library feature `rustc_contracts_internals`
    core::contracts::build_check_ensures(|_: &()| true);
    //~^ ERROR use of unstable library feature `rustc_contracts_internals`

    // ast extensions are guarded by rustc_contracts_internals feature gate
    fn identity_1() -> i32 rustc_contract_requires(|| true) { 10 }
    //~^ ERROR contract internal machinery is for internal use only
    fn identity_2() -> i32 rustc_contract_ensures(|_| true) { 10 }
    //~^ ERROR contract internal machinery is for internal use only
}
