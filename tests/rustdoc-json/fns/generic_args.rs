//@ set foo = "$.index[?(@.name=='Foo')].id"
pub trait Foo {}

//@ set generic_foo = "$.index[?(@.name=='GenericFoo')].id"
pub trait GenericFoo<'a> {}

//@ is "$.index[?(@.name=='generics')].inner.function.generics.where_predicates" "[]"
//@ count "$.index[?(@.name=='generics')].inner.function.generics.params[*]" 1
//@ is "$.index[?(@.name=='generics')].inner.function.generics.params[0].name" '"F"'
//@ is "$.index[?(@.name=='generics')].inner.function.generics.params[0].kind.type.default" 'null'
//@ count "$.index[?(@.name=='generics')].inner.function.generics.params[0].kind.type.bounds[*]" 1
//@ is "$.index[?(@.name=='generics')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" '$foo'
//@ has "$.index[?(@.name=='generics')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='generics')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='generics')].inner.function.sig.inputs[*]" 1
//@ is "$.index[?(@.name=='generics')].inner.function.sig.inputs[0][0]" '"f"'
//@ is "$.index[?(@.name=='generics')].inner.function.sig.inputs[0][1].generic" '"F"'
pub fn generics<F: Foo>(f: F) {}

//@ is "$.index[?(@.name=='impl_trait')].inner.function.generics.where_predicates" "[]"
//@ count "$.index[?(@.name=='impl_trait')].inner.function.generics.params[*]" 1
//@ is "$.index[?(@.name=='impl_trait')].inner.function.generics.params[0].name" '"impl Foo"'
//@ is "$.index[?(@.name=='impl_trait')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $foo
//@ has "$.index[?(@.name=='impl_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='impl_trait')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[*]" 1
//@ is "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][0]" '"f"'
//@ count "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][1].impl_trait.bounds[*]" 1
//@ is "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][1].impl_trait.bounds[0].trait_bound.trait.id" $foo
//@ has "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='impl_trait')].inner.function.sig.inputs[0][1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Foo')]"
pub fn impl_trait(f: impl Foo) {}

//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.params[*]" 3
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].name" '"F"'
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.bounds" "[]"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.default" 'null'
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.is_synthetic" 'false'
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Foo')]"
//@ has "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[1].name" '"G"'
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='GenericFoo')]"
//@ has "$.index[?(@.name=='where_clase')].inner.function.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[1].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.params[2].name" '"H"'
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Foo')]"
//@ has "$.index[?(@.name=='where_clase')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
//@ !has "$.index[?(@.name=='where_clase')].inner.function.generics.params[2].kind.type.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='where_clase')].inner.function.sig.inputs[*]" 3
//@ is "$.index[?(@.name=='where_clase')].inner.function.sig.inputs[0][0]" '"f"'
//@ is "$.index[?(@.name=='where_clase')].inner.function.sig.inputs[0][1].generic" '"F"'
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[*]" 3

//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[0].bound_predicate.type.generic" \"F\"
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[0].bound_predicate.bounds[*]" 1
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.id" $foo
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.type.generic" \"G\"
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.bounds[*]" 1
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.trait.id" $generic_foo
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[*]" 1
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[0].name" \"\'a\"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.bounds[0].trait_bound.generic_params[0].kind.lifetime.outlives" "[]"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[1].bound_predicate.generic_params" "[]"

//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.type.borrowed_ref.lifetime" \"\'b\"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.type.borrowed_ref.type.generic" \"H\"
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.bounds[*]" 1
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.bounds[0].trait_bound.trait.id" $foo
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.bounds[0].trait_bound.generic_params" "[]"
//@ count "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.generic_params[*]" 1
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.generic_params[0].name" \"\'b\"
//@ is "$.index[?(@.name=='where_clase')].inner.function.generics.where_predicates[2].bound_predicate.generic_params[0].kind.lifetime.outlives" "[]"
pub fn where_clase<F, G, H>(f: F, g: G, h: H)
where
    F: Foo,
    G: for<'a> GenericFoo<'a>,
    for<'b> &'b H: Foo,
{
}
