// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @set foo = generic_args.json "$.index[*][?(@.name=='Foo')].id"
pub trait Foo {}

// @is - "$.index[*][?(@.name=='generics')].inner.generics.where_predicates" "[]"
// @count - "$.index[*][?(@.name=='generics')].inner.generics.params[*]" 1
// @is - "$.index[*][?(@.name=='generics')].inner.generics.params[0].name" '"F"'
// @is - "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.default" 'null'
// @count - "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.bounds[*]" 1
// @is - "$.index[*][?(@.name=='generics')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.inner.id" '$foo'
// @count - "$.index[*][?(@.name=='generics')].inner.decl.inputs[*]" 1
// @is - "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][0]" '"f"'
// @is - "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][1].kind" '"generic"'
// @is - "$.index[*][?(@.name=='generics')].inner.decl.inputs[0][1].inner" '"F"'
pub fn generics<F: Foo>(f: F) {}

// @is - "$.index[*][?(@.name=='impl_trait')].inner.generics.where_predicates" "[]"
// @count - "$.index[*][?(@.name=='impl_trait')].inner.generics.params[*]" 1
// @is - "$.index[*][?(@.name=='impl_trait')].inner.generics.params[0].name" '"impl Foo"'
// @is - "$.index[*][?(@.name=='impl_trait')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.inner.id" $foo
// @count - "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[*]" 1
// @is - "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][0]" '"f"'
// @is - "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].kind" '"impl_trait"'
// @count - "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].inner[*]" 1
// @is - "$.index[*][?(@.name=='impl_trait')].inner.decl.inputs[0][1].inner[0].trait_bound.trait.inner.id" $foo
pub fn impl_trait(f: impl Foo) {}

// @count - "$.index[*][?(@.name=='where_clase')].inner.generics.params[*]" 1
// @is - "$.index[*][?(@.name=='where_clase')].inner.generics.params[0].name" '"F"'
// @is - "$.index[*][?(@.name=='where_clase')].inner.generics.params[0].kind" '{"type": {"bounds": [], "default": null, "synthetic": false}}'
// @count - "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[*]" 1
// @is - "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][0]" '"f"'
// @is - "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][1].kind" '"generic"'
// @is - "$.index[*][?(@.name=='where_clase')].inner.decl.inputs[0][1].inner" '"F"'
// @count - "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[*]" 1
// @is - "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.type" '{"inner": "F", "kind": "generic"}'
// @count - "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.bounds[*]" 1
// @is - "$.index[*][?(@.name=='where_clase')].inner.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.inner.id" $foo
pub fn where_clase<F>(f: F)
where
    F: Foo,
{
}
