//@ revisions: unchk_pass unchk_fail_pre unchk_fail_post chk_pass chk_fail_pre chk_fail_post
//
//@ [unchk_pass] run-pass
//@ [unchk_fail_pre] run-pass
//@ [unchk_fail_post] run-pass
//@ [chk_pass] run-pass
//
//@ [chk_fail_pre] run-crash
//@ [chk_fail_post] run-crash
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [unchk_fail_pre] compile-flags: -Zcontract-checks=no
//@ [unchk_fail_post] compile-flags: -Zcontract-checks=no
//
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_pre] compile-flags: -Zcontract-checks=yes
//@ [chk_fail_post] compile-flags: -Zcontract-checks=yes

#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

#[core::contracts::requires(x.baz > 0)]
#[core::contracts::ensures(|ret| *ret > 100)]
fn tail(x: Baz) -> i32
{
    x.baz + 50
}

struct Baz { baz: i32 }

const BAZ_PASS_PRE_POST: Baz = Baz { baz: 100 };
#[cfg(any(unchk_fail_post, chk_fail_post))]
const BAZ_FAIL_POST: Baz = Baz { baz: 10 };
#[cfg(any(unchk_fail_pre, chk_fail_pre))]
const BAZ_FAIL_PRE: Baz = Baz { baz: -10 };

fn main() {
    assert_eq!(tail(BAZ_PASS_PRE_POST), 150);
    #[cfg(any(unchk_fail_pre, chk_fail_pre))]
    tail(BAZ_FAIL_PRE);
    #[cfg(any(unchk_fail_post, chk_fail_post))]
    tail(BAZ_FAIL_POST);
}
