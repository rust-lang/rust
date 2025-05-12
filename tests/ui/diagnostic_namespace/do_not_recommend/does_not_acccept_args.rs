//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.syntax

trait Foo {}
trait Bar {}
trait Baz {}

#[diagnostic::do_not_recommend(not_accepted)]
//~^ WARNING `#[diagnostic::do_not_recommend]` does not expect any arguments
impl<T> Foo for T where T: Send {}

#[diagnostic::do_not_recommend(not_accepted = "foo")]
//~^ WARNING `#[diagnostic::do_not_recommend]` does not expect any arguments
impl<T> Bar for T where T: Send {}

#[diagnostic::do_not_recommend(not_accepted(42))]
//~^ WARNING `#[diagnostic::do_not_recommend]` does not expect any arguments
impl<T> Baz for T where T: Send {}

fn main() {}
