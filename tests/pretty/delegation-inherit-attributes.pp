//@ edition:2021
//@ aux-crate:to_reuse_functions=to-reuse-functions.rs
//@ pretty-mode:hir
//@ pretty-compare-only
//@ pp-exact:delegation-inherit-attributes.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]
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

mod recursive {
    // Check that `baz` inherit attribute from `foo`
    mod first {
        fn bar() { }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo() -> _ { bar() }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn baz() -> _ { foo() }
    }

    // Check that `baz` inherit attribute from `bar`
    mod second {
        #[attr = MustUse {reason: "some reason"}]
        fn bar() { }

        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo() -> _ { bar() }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn baz() -> _ { foo() }
    }

    // Check that `foo5` don't inherit attribute from `bar`
    // and inherit attribute from foo4, check that foo1, foo2 and foo3
    // inherit attribute from bar
    mod third {
        #[attr = MustUse {reason: "some reason"}]
        fn bar() { }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo1() -> _ { bar() }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo2() -> _ { foo1() }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo3() -> _ { foo2() }
        #[attr = MustUse {reason: "foo4"}]
        #[attr = Inline(Hint)]
        fn foo4() -> _ { foo3() }
        #[attr = MustUse {reason: "foo4"}]
        #[attr = Inline(Hint)]
        fn foo5() -> _ { foo4() }
    }

    mod fourth {
        trait T {
            fn foo(&self, x: usize) -> usize { x + 1 }
        }

        struct X;
        impl T for X { }

        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn foo(self: _, arg1: _) -> _ { <X as T>::foo(self + 1, arg1) }
        #[attr = MustUse {reason: "some reason"}]
        #[attr = Inline(Hint)]
        fn bar(self: _, arg1: _) -> _ { foo(self + 1, arg1) }
    }
}

fn main() { }
