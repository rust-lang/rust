//@ run-pass
#![feature(contracts, cfg_contract_checks, contracts_internals, core_intrinsics)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

extern crate core;

// we check here if the "lowered" program behaves as expected before
// implementing the actual lowering in the compiler

fn foo(x: u32) -> u32 {
    let post = {
        let y = 2 * x;
        // call contract_check_requires here to avoid borrow checker issues
        // with variables declared in contract requires
        core::intrinsics::contract_check_requires(|| y > 0);
        Some(core::contracts::build_check_ensures(move |ret| *ret == y))
    };

    core::intrinsics::contract_check_ensures(post, { 2 * x })
}

fn main() {
    foo(1);
}
