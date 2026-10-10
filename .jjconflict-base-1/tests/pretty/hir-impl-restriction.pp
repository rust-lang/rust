#![attr = Feature([impl_restriction#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-impl-restriction.pp

impl(in crate) trait T1 { }
impl(in self) trait T2 { }

mod a {
    impl(in crate::a) trait T3 { }
    impl(in super) trait T4 { }
}
