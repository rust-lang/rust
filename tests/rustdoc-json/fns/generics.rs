//@ set wham_id = "$.index[?(@.name=='Wham')].id"
pub trait Wham {}

//@ is    "$.index[?(@.name=='one_generic_param_fn')].inner.function.generics.where_predicates" []
//@ count "$.index[?(@.name=='one_generic_param_fn')].inner.function.generics.params[*]" 1
//@ is    "$.index[?(@.name=='one_generic_param_fn')].inner.function.generics.params[0].name" '"T"'
//@ is    "$.index[?(@.name=='one_generic_param_fn')].inner.function.generics.params[0].kind.type.is_synthetic" false
//@ is    "$.index[?(@.name=='one_generic_param_fn')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $wham_id
//@ is    "$.index[?(@.name=='one_generic_param_fn')].inner.function.sig.inputs" '[["w", {"generic": "T"}]]'
pub fn one_generic_param_fn<T: Wham>(w: T) {}

//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.generics.where_predicates" []
//@ count "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.generics.params[*]" 1
//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.generics.params[0].name" '"impl Wham"'
//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.generics.params[0].kind.type.is_synthetic" true
//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.id" $wham_id
//@ count "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.sig.inputs[*]" 1
//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.sig.inputs[0][0]" '"w"'
//@ is    "$.index[?(@.name=='one_synthetic_generic_param_fn')].inner.function.sig.inputs[0][1].impl_trait[0].trait_bound.trait.id" $wham_id
pub fn one_synthetic_generic_param_fn(w: impl Wham) {}
