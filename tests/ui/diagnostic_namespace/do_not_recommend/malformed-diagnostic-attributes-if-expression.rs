//@ check-pass
//@ reference: attributes.diagnostic.do_not_recommend.syntax

trait Foo {}

#[diagnostic::do_not_recommend(if not_accepted)]
//~^ WARNING `#[diagnostic::do_not_recommend]` does not expect any arguments
impl Foo for () {}

fn main() {}
