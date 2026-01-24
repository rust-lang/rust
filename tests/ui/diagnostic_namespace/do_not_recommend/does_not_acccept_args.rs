//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.syntax

trait Foo {}
trait Bar {}
trait Baz {}
trait Boo {}

#[diagnostic::do_not_recommend(not_accepted)]
//~^ ERROR  malformed `diagnostic::do_not_recommend` attribute input
//~| NOTE didn't expect any arguments here
impl<T> Foo for T where T: Send {}

#[diagnostic::do_not_recommend(not_accepted = "foo")]
//~^ ERROR  malformed `diagnostic::do_not_recommend` attribute input
//~| NOTE didn't expect any arguments here
impl<T> Bar for T where T: Send {}

#[diagnostic::do_not_recommend(not_accepted(42))]
//~^ ERROR  malformed `diagnostic::do_not_recommend` attribute input
//~| NOTE didn't expect any arguments here
impl<T> Baz for T where T: Send {}

#[diagnostic::do_not_recommend(x = y + z)]
//~^ ERROR  malformed `diagnostic::do_not_recommend` attribute input
//~| NOTE didn't expect any arguments here
impl<T> Boo for T where T: Send {}

fn main() {}
