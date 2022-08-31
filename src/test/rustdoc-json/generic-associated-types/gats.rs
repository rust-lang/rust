// ignore-tidy-linelength

#![no_core]
#![feature(generic_associated_types, lang_items, no_core)]

#[lang = "sized"]
pub trait Sized {}

pub trait Display {}

pub trait LendingIterator {
    // @count "$.index[*][?(@.name=='LendingItem')].inner.generics.params[*]" 1
    // @is "$.index[*][?(@.name=='LendingItem')].inner.generics.params[*].name" \"\'a\"
    // @count "$.index[*][?(@.name=='LendingItem')].inner.generics.where_predicates[*]" 1
    // @is "$.index[*][?(@.name=='LendingItem')].inner.generics.where_predicates[*].bound_predicate.type.inner" \"Self\"
    // @is "$.index[*][?(@.name=='LendingItem')].inner.generics.where_predicates[*].bound_predicate.bounds[*].outlives" \"\'a\"
    // @count "$.index[*][?(@.name=='LendingItem')].inner.bounds[*]" 1
    type LendingItem<'a>: Display
    where
        Self: 'a;

    // @is "$.index[*][?(@.name=='lending_next')].inner.decl.output.kind" \"qualified_path\"
    // @count "$.index[*][?(@.name=='lending_next')].inner.decl.output.inner.args.angle_bracketed.args[*]" 1
    // @count "$.index[*][?(@.name=='lending_next')].inner.decl.output.inner.args.angle_bracketed.bindings[*]" 0
    // @is "$.index[*][?(@.name=='lending_next')].inner.decl.output.inner.self_type.inner" \"Self\"
    // @is "$.index[*][?(@.name=='lending_next')].inner.decl.output.inner.name" \"LendingItem\"
    fn lending_next<'a>(&'a self) -> Self::LendingItem<'a>;
}

pub trait Iterator {
    // @count "$.index[*][?(@.name=='Item')].inner.generics.params[*]" 0
    // @count "$.index[*][?(@.name=='Item')].inner.generics.where_predicates[*]" 0
    // @count "$.index[*][?(@.name=='Item')].inner.bounds[*]" 1
    type Item: Display;

    // @is "$.index[*][?(@.name=='next')].inner.decl.output.kind" \"qualified_path\"
    // @count "$.index[*][?(@.name=='next')].inner.decl.output.inner.args.angle_bracketed.args[*]" 0
    // @count "$.index[*][?(@.name=='next')].inner.decl.output.inner.args.angle_bracketed.bindings[*]" 0
    // @is "$.index[*][?(@.name=='next')].inner.decl.output.inner.self_type.inner" \"Self\"
    // @is "$.index[*][?(@.name=='next')].inner.decl.output.inner.name" \"Item\"
    fn next<'a>(&'a self) -> Self::Item;
}
