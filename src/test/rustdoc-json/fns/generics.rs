// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

// @set wham_id = "$.index[*][?(@.name=='Wham')].id"
pub trait Wham {}

// @is    "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.where_predicates" []
// @count "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.params[*]" 1
// @is    "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.params[0].name" '"T"'
// @has   "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.params[0].kind.type.synthetic" false
// @has   "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $wham_id
// @is    "$.index[*][?(@.name=='one_generic_param_fn')].inner.decl.inputs" '[["w", {"inner": "T", "kind": "generic"}]]'
pub fn one_generic_param_fn<T: Wham>(w: T) {}

// @is    "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.where_predicates" []
// @count "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.params[*]" 1
// @is    "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.params[0].name" '"impl Wham"'
// @has   "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.params[0].kind.type.synthetic" true
// @has   "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $wham_id
// @count "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.decl.inputs[*]" 1
// @is    "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.decl.inputs[0][0]" '"w"'
// @is    "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.decl.inputs[0][1].kind" '"impl_trait"'
// @is    "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.decl.inputs[0][1].inner[0].trait_bound.trait.id" $wham_id
pub fn one_synthetic_generic_param_fn(w: impl Wham) {}
