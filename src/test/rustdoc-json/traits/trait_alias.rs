// Regression test for <https://github.com/rust-lang/rust/issues/104923>
// ignore-tidy-linelength

#![feature(trait_alias)]

// @set Orig = "$.index[*][?(@.name == 'Orig')].id"
// @is "$.index[*][?(@.name == 'Orig')].kind" '"trait"'
pub trait Orig<T> {}

// @set Alias = "$.index[*][?(@.name == 'Alias')].id"
// @is "$.index[*][?(@.name == 'Alias')].kind" '"trait_alias"'
// @is "$.index[*][?(@.name == 'Alias')].inner.generics" '{"params": [], "where_predicates": []}'
// @count "$.index[*][?(@.name == 'Alias')].inner.params[*]" 1
// @is "$.index[*][?(@.name == 'Alias')].inner.params[0].trait_bound.trait.id" $Orig
// @is "$.index[*][?(@.name == 'Alias')].inner.params[0].trait_bound.trait.args.angle_bracketed.args[0].type.inner" '"i32"'
pub trait Alias = Orig<i32>;

pub struct Struct;

impl Orig<i32> for Struct {}

// @is "$.index[*][?(@.name=='takes_alias')].inner.decl.inputs[0][1].kind" '"impl_trait"'
// @is "$.index[*][?(@.name=='takes_alias')].inner.decl.inputs[0][1].inner[0].trait_bound.trait.id" $Alias
// @is "$.index[*][?(@.name=='takes_alias')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $Alias
pub fn takes_alias(_: impl Alias) {}
// FIXME: Should the trait be mentioned in both the decl and generics?

fn main() {
    takes_alias(Struct);
}
