// Test-pass variant of #69136

// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait SomeTrait {}

impl SomeTrait for () {}

trait WithAssoc {
    type AssocType;
}

impl WithAssoc for () {
    type AssocType = ();
}

type Return<'a> = impl WithAssoc<AssocType = impl Sized + 'a>;

fn my_fun<'a>() -> Return<'a> {}

fn main() {}
