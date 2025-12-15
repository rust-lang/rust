//@ edition:2021
//@ aux-crate:to_reuse_functions=to-reuse-functions.rs
//@ pretty-mode:hir
//@ pretty-compare-only
//@ pp-exact:delegation-inherit-attributes.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]
#[attr = MacroUse {arguments: UseAll}]
extern crate std;
#[prelude_import]
use std::prelude::rust_2021::*;

extern crate to_reuse_functions;

mod to_reuse {
    #[attr = MustUse {reason: "foo: some reason"}]
    #[attr = Cold]
    fn foo(x: usize) -> usize { x }

    #[attr = MustUse]
    #[attr = Cold]
    fn foo_no_reason(x: usize) -> usize { x }

    #[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
    #[attr = Cold]
    fn bar(x: usize) -> usize { x }
}

#[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
#[attr = MustUse {reason: "foo: some reason"}]
#[attr = Inline(Hint)]
fn foo1(arg0: _) -> _ { to_reuse::foo(self + 1) }

#[attr = MustUse]
#[attr = Inline(Hint)]
fn foo_no_reason(arg0: _) -> _ { to_reuse::foo_no_reason(self + 1) }

#[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
#[attr = MustUse {reason: "some reason"}]
#[attr = Inline(Hint)]
fn foo2(arg0: _) -> _ { to_reuse::foo(self + 1) }

#[attr = Inline(Hint)]
fn bar(arg0: _) -> _ { to_reuse::bar(arg0) }

#[attr = MustUse]
#[attr = Inline(Hint)]
unsafe fn unsafe_fn_extern() -> _ { to_reuse_functions::unsafe_fn_extern() }
#[attr = MustUse {reason: "extern_fn_extern: some reason"}]
#[attr = Inline(Hint)]
extern "C" fn extern_fn_extern()
    -> _ { to_reuse_functions::extern_fn_extern() }
#[attr = Inline(Hint)]
const fn const_fn_extern() -> _ { to_reuse_functions::const_fn_extern() }
#[attr = MustUse {reason: "some reason"}]
#[attr = Inline(Hint)]
async fn async_fn_extern() -> _ { to_reuse_functions::async_fn_extern() }


fn main() { }
