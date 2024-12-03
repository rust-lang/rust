//@ revisions: unchk_pass unchk_fail_pre unchk_fail_post chk_pass chk_fail_pre chk_fail_post
//
//@ [unchk_pass] run-pass
//@ [unchk_fail_pre] run-pass
//@ [unchk_fail_post] run-pass
//@ [chk_pass] run-pass
//
//@ [chk_fail_pre] run-fail
//@ [chk_fail_post] run-fail
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [unchk_fail_pre] compile-flags: -Zcontract-checks=no
//@ [unchk_fail_post] compile-flags: -Zcontract-checks=no
//
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_pre] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_post] compile-flags: -Zcontract-checks=yes

#![feature(rustc_contracts)] // to access core::contracts
#![feature(rustc_contracts_internals)] // to access check_requires lang item

fn foo(x: Baz) -> i32 {
    core::contracts::check_requires(|| x.baz > 0);

    let injected_checker = {
        core::contracts::build_check_ensures(|ret| *ret > 100)
    };

    let ret = x.baz + 50;
    injected_checker(ret)
}

struct Baz { baz: i32 }


const BAZ_PASS_PRE_POST: Baz = Baz { baz: 100 };
#[cfg(any(unchk_fail_post, chk_fail_post))]
const BAZ_FAIL_POST: Baz = Baz { baz: 10 };
#[cfg(any(unchk_fail_pre, chk_fail_pre))]
const BAZ_FAIL_PRE: Baz = Baz { baz: -10 };

fn main() {
    assert_eq!(foo(BAZ_PASS_PRE_POST), 150);
    #[cfg(any(unchk_fail_pre, chk_fail_pre))]
    foo(BAZ_FAIL_PRE);
    #[cfg(any(unchk_fail_post, chk_fail_post))]
    foo(BAZ_FAIL_POST);
}
