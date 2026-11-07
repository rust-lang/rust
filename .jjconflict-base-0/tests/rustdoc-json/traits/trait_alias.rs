// Regression test for <https://github.com/rust-lang/rust/issues/104923>

#![feature(trait_alias)]

//@ set Orig = "$.index[?(@.name == 'Orig')].id"
//@ has "$.index[?(@.name == 'Orig')].inner.trait"
pub trait Orig<T> {}

//@ set Alias = "$.index[?(@.name == 'Alias')].id"
//@ has "$.index[?(@.name == 'Alias')].inner.trait_alias"
//@ is "$.index[?(@.name == 'Alias')].inner.trait_alias.generics" '{"params": [], "where_predicates": []}'
//@ count "$.index[?(@.name == 'Alias')].inner.trait_alias.params[*]" 1
//@ is "$.index[?(@.name == 'Alias')].inner.trait_alias.params[0].trait_bound.trait.id" $Orig
//@ is "$.index[?(@.name == 'Alias')].inner.trait_alias.params[0].trait_bound.trait.args.angle_bracketed.args[0].type.primitive" '"i32"'
pub trait Alias = Orig<i32>;

pub struct Struct;

impl Orig<i32> for Struct {}

//@ has "$.index[?(@.name=='takes_alias')].inner.function.sig.inputs[0][1].impl_trait"
//@ is "$.index[?(@.name=='takes_alias')].inner.function.sig.inputs[0][1].impl_trait[0].trait_bound.trait.id" $Alias
//@ is "$.index[?(@.name=='takes_alias')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $Alias
pub fn takes_alias(_: impl Alias) {}
// FIXME: Should the trait be mentioned in both the decl and generics?

fn main() {
    takes_alias(Struct);
}
