#![feature(impl_restriction)]
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-impl-restriction.pp

impl(crate) trait T1 {}
impl(self) trait T2 {}

mod a {
    impl(in crate::a) trait T3 {}
    impl(super) trait T4 {}
}
