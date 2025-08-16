//@ revisions: chk_pass chk_fail_pre chk_fail_post
//
//@ [chk_pass] run-pass
//
//@ [chk_fail_pre] run-crash
//@ [chk_fail_post] run-crash

#![feature(contracts_internals)]

fn tail(x: Baz) -> i32
    contract_requires(|| x.baz > 0)
    contract_ensures(|ret| *ret > 100)
{
    x.baz + 50
}

struct Baz { baz: i32 }

const BAZ_PASS_PRE_POST: Baz = Baz { baz: 100 };
#[cfg(chk_fail_post)]
const BAZ_FAIL_POST: Baz = Baz { baz: 10 };
#[cfg(chk_fail_pre)]
const BAZ_FAIL_PRE: Baz = Baz { baz: -10 };

fn main() {
    assert_eq!(tail(BAZ_PASS_PRE_POST), 150);
    #[cfg(chk_fail_pre)]
    tail(BAZ_FAIL_PRE);
    #[cfg(chk_fail_post)]
    tail(BAZ_FAIL_POST);
}
