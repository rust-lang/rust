//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ reference: attributes.diagnostic.do_not_recommend.intro

trait Foo {}

#[diagnostic::do_not_recommend]
impl<T> Foo for T where T: Send {}

fn needs_foo<T: Foo>() {}
//~^ NOTE required by a bound in `needs_foo`
//~| NOTE required by this bound in `needs_foo`

fn main() {
    needs_foo::<*mut ()>();
    //~^ ERROR the trait bound `*mut (): Foo` is not satisfied
    //~| NOTE the trait `Foo` is not implemented for `*mut ()`
}
