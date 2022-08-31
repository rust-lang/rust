// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @set foo = "$.index[*][?(@.name=='Foo')].id"
pub trait Foo {}

// @set generic_foo = "$.index[*][?(@.name=='GenericFoo')].id"
pub trait GenericFoo<'a> {}

// @is "$.index[*][?(@.name=='generics')].inner.generics.where_predicates" "[]"
// @count "$.index[*][?(@.name=='generics')].inner.generics.params[*]" 1
// @is "$.index[*][?(@.name=='generics')].inner.generics.params[0].name" '"F"'
// @is "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.default" 'null'
// @count "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.bounds[*]" 1
// @is "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" '$foo'
// @count "$.index[*][?(@.name=='generics')].inner.decl.inputs[*]" 1
// @is "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][0]" '"f"'
// @is "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][1].kind" '"generic"'
// @is "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][1].inner" '"F"'
pub fn generics<F: Foo>(f: F) {}

// @is "$.index[*][?(@.name=='impl_trait')].inner.generics.where_predicates" "[]"
// @count "$.index[*][?(@.name=='impl_trait')].inner.generics.params[*]" 1
// @is "$.index[*][?(@.name=='impl_trait')].inner.generics.params[0].name" '"impl Foo"'
// @is "$.index[*][?(@.name=='impl_trait')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $foo
// @count "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[*]" 1
// @is "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][0]" '"f"'
// @is "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].kind" '"impl_trait"'
// @count "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].inner[*]" 1
// @is "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].inner[0].trait_bound.trait.id" $foo
pub fn impl_trait(f: impl Foo) {}

// @count "$.index[*][?(@.name=='where_clase')].inner.generics.params[*]" 3
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.params[0].name" '"F"'
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.params[0].kind" '{"type": {"bounds": [], "default": null, "synthetic": false}}'
// @count "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[*]" 3
// @is "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][0]" '"f"'
// @is "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][1].kind" '"generic"'
// @is "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][1].inner" '"F"'
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[*]" 3

// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.type" '{"inner": "F", "kind": "generic"}'
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.bounds[*]" 1
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.id" $foo

// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.type" '{"inner": "G", "kind": "generic"}'
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.bounds[*]" 1
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.trait.id" $generic_foo
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[*]" 1
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[0].name" \"\'a\"
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[1].bound_predicate.generic_params" "[]"

// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.type.kind" '"borrowed_ref"'
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.type.inner.lifetime" \"\'b\"
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.type.inner.type" '{"inner": "H", "kind": "generic"}'
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.bounds[*]" 1
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.bounds[0].trait_bound.trait.id" $foo
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.bounds[0].trait_bound.generic_params" "[]"
// @count "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.generic_params[*]" 1
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.generic_params[0].name" \"\'b\"
// @is "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[2].bound_predicate.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
pub fn where_clase<F, G, H>(f: F, g: G, h: H)
where
    F: Foo,
    G: for<'a> GenericFoo<'a>,
    for<'b> &'b H: Foo,
{
}
