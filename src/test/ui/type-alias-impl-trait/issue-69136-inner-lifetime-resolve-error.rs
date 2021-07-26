// Regression test for #69136

#![feature(type_alias_impl_trait)]

trait SomeTrait {}

impl SomeTrait for () {}

trait WithAssoc<A> {
    type AssocType;
}

impl<T> WithAssoc<T> for () {
    type AssocType = ();
}

type Return<A> = impl WithAssoc<A, AssocType = impl SomeTrait + 'a>;
//~^ ERROR use of undeclared lifetime name `'a`

fn my_fun() -> Return<()> {}

fn main() {}
