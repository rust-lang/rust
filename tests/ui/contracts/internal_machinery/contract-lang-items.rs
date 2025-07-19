//@ revisions: unchk_pass unchk_fail_post chk_pass chk_fail_post
//
//@ [unchk_pass] run-pass
//@ [unchk_fail_post] run-pass
//@ [chk_pass] run-pass
//
//@ [chk_fail_post] run-crash
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [unchk_fail_post] compile-flags: -Zcontract-checks=no
//
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_post] compile-flags: -Zcontract-checks=yes

#![feature(contracts)] // to access core::contracts
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]
#![feature(contracts_internals)] // to access check_requires lang item
#![feature(core_intrinsics)]
fn foo(x: Baz) -> i32 {
    let injected_checker = {
        core::contracts::build_check_ensures(|ret| *ret > 100)
    };

    let ret = x.baz + 50;
    core::intrinsics::contract_check_ensures(injected_checker, ret)
}

struct Baz { baz: i32 }


const BAZ_PASS_PRE_POST: Baz = Baz { baz: 100 };
#[cfg(any(unchk_fail_post, chk_fail_post))]
const BAZ_FAIL_POST: Baz = Baz { baz: 10 };

fn main() {
    assert_eq!(foo(BAZ_PASS_PRE_POST), 150);
    #[cfg(any(unchk_fail_post, chk_fail_post))]
    foo(BAZ_FAIL_POST);
}
