// Test-pass variant of #69136

//@ check-pass

#![feature(type_alias_impl_trait)]

trait SomeTrait {}

impl SomeTrait for () {}

trait WithAssoc {
    type AssocType;
}

impl WithAssoc for () {
    type AssocType = ();
}

type Return<'a> = impl WithAssoc<AssocType = impl Sized + 'a>;

#[define_opaque(Return)]
fn my_fun<'a>() -> Return<'a> {}

fn main() {}
