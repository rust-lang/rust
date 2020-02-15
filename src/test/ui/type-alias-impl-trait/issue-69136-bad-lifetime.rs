// Regression test for issue #69136
// Ensures that we do not ICE after emitting an error
// for an unresolved region in a nested TAIT.

#![feature(type_alias_impl_trait)]

trait SomeTrait {}

trait WithAssoc<A> {
    type AssocType;
}

type Return<A> = impl WithAssoc<A, AssocType = impl SomeTrait + 'a>;
//~^ ERROR use of undeclared lifetime name `'a`

fn my_fun() -> Return<()> {}

fn main() {}
