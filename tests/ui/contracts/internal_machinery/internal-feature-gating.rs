// gate-test-contracts_internals

fn main() {
    // intrinsics are guarded by contracts_internals feature gate.
    core::intrinsics::contract_checks();
    //~^ ERROR use of unstable library feature `contracts_internals`
    core::intrinsics::contract_check_requires(|| true);
    //~^ ERROR use of unstable library feature `contracts_internals`
    core::intrinsics::contract_check_ensures( |_|true, &1);
    //~^ ERROR use of unstable library feature `contracts_internals`

    core::contracts::build_check_ensures(|_: &()| true);
    //~^ ERROR use of unstable library feature `contracts_internals`

    // ast extensions are guarded by contracts_internals feature gate
    fn identity_1() -> i32 contract_requires(|| true) { 10 }
    //~^ ERROR contract internal machinery is for internal use only
    fn identity_2() -> i32 contract_ensures(|_| true) { 10 }
    //~^ ERROR contract internal machinery is for internal use only
}
