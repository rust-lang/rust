//@ revisions: unchk_pass chk_pass chk_fail_post
//
//@ [unchk_pass] run-pass
//@ [chk_pass] run-pass
//
//@ [chk_fail_post] run-crash

#![expect(incomplete_features)]
#![feature(contracts)] // to access core::contracts
#![feature(contracts_internals)] // to access check_requires lang item
#![feature(core_intrinsics)]
fn foo(x: Baz) -> i32 {
    let injected_checker =  Some(core::contracts::build_check_ensures(|ret| *ret > 100));

    let ret = x.baz + 50;
    core::intrinsics::contract_check_ensures(injected_checker, ret)
}

struct Baz { baz: i32 }


const BAZ_PASS_PRE_POST: Baz = Baz { baz: 100 };
#[cfg(chk_fail_post)]
const BAZ_FAIL_POST: Baz = Baz { baz: 10 };

fn main() {
    assert_eq!(foo(BAZ_PASS_PRE_POST), 150);
    #[cfg(chk_fail_post)]
    foo(BAZ_FAIL_POST);
}
