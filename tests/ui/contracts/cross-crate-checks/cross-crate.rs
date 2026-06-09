//@ aux-build:id.rs
//@ revisions: unchk_pass unchk_fail chk_pass chk_fail
//
// The dependency crate `id` can be compiled with runtime contract checking
// enabled independently of whether this crate is compiled with contract checks
// or not.
//
// chk/unchk indicates whether this crate is compiled with contracts or not
// and pass/fail indicates whether the `id` crate is compiled with contract checks.
//
//@ [unchk_pass] run-pass
//@ [unchk_fail] run-crash
//@ [chk_pass] run-pass
//@ [chk_fail] run-crash
//
//
//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [unchk_fail] compile-flags: -Zcontract-checks=no
//@ [chk_pass] compile-flags: -Zcontract-checks=yes
//@ [chk_fail] compile-flags: -Zcontract-checks=yes

extern crate id;

fn main() {
    id::id_if_positive(0);
}
