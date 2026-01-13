//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:delegation-inline-attribute.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

mod to_reuse {
    fn foo(x: usize) -> usize { x }
}

// Check that #[inline(hint)] is added to foo reuse
#[attr = Inline(Hint)]
fn bar(arg0: _) -> _ { to_reuse::foo(self + 1) }

trait Trait {
    fn foo(&self) { }
    fn foo1(&self) { }
    fn foo2(&self) { }
    fn foo3(&self) { }
    fn foo4(&self) { }
}

impl Trait for u8 { }

struct S(u8);

mod to_import {
    fn check(arg: &'_ u8) -> &'_ u8 { arg }
}

impl Trait for S {
    // Check that #[inline(hint)] is added to foo reuse
    #[attr = Inline(Hint)]
    fn foo(self: _)
        ->
            _ {
        {
                // Check that #[inline(hint)] is added to foo0 reuse inside another reuse
                #[attr = Inline(Hint)]
                fn foo0(arg0: _) -> _ { to_reuse::foo(self + 1) }

                // Check that #[inline(hint)] is added when other attributes present in inner reuse
                #[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
                #[attr = MustUse]
                #[attr = Cold]
                #[attr = Inline(Hint)]
                fn foo1(arg0: _) -> _ { to_reuse::foo(self / 2) }

                // Check that #[inline(never)] is preserved in inner reuse
                #[attr = Inline(Never)]
                fn foo2(arg0: _) -> _ { to_reuse::foo(self / 2) }

                // Check that #[inline(always)] is preserved in inner reuse
                #[attr = Inline(Always)]
                fn foo3(arg0: _) -> _ { to_reuse::foo(self / 2) }

                // Check that #[inline(never)] is preserved when there are other attributes in inner reuse
                #[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
                #[attr = Inline(Never)]
                #[attr = MustUse]
                #[attr = Cold]
                fn foo4(arg0: _) -> _ { to_reuse::foo(self / 2) }
            }.foo()
    }

    // Check that #[inline(hint)] is added when there are other attributes present in trait reuse
    #[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
    #[attr = MustUse]
    #[attr = Cold]
    #[attr = Inline(Hint)]
    fn foo1(self: _) -> _ { self.0.foo1() }

    // Check that #[inline(never)] is preserved in trait reuse
    #[attr = Inline(Never)]
    fn foo2(self: _) -> _ { self.0.foo2() }

    // Check that #[inline(always)] is preserved in trait reuse
    #[attr = Inline(Always)]
    fn foo3(self: _) -> _ { self.0.foo3() }

    // Check that #[inline(never)] is preserved when there are other attributes in trait reuse
    #[attr = Deprecation {deprecation: Deprecation {since: Unspecified}}]
    #[attr = Inline(Never)]
    #[attr = MustUse]
    #[attr = Cold]
    fn foo4(self: _) -> _ { self.0.foo4() }
}

fn main() { }
