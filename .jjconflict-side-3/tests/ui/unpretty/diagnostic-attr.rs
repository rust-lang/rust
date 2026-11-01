//@ compile-flags: -Zunpretty=hir
//@ check-pass
//@ edition: 2015
#![feature(diagnostic_on_const)]

#[diagnostic::on_unimplemented(
    message = "My Message for `ImportantTrait<{A}>` implemented for `{Self}`",
    label = "My Label",
    note = "Note 1",
    note = "Note 2"
)]
pub trait ImportantTrait<A> {}

#[diagnostic::do_not_recommend]
impl<T> ImportantTrait<T> for T where T: Clone {}

pub struct X;

#[diagnostic::on_const(
    message = "My Message for `ImportantTrait<u8>` implemented for `{Self}`",
    label = "My Label",
    note = "Note 1",
    note = "Note 2"
)]
impl ImportantTrait<u8> for X {}
