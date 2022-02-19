// @has generics.json "$.index[*][?(@.name=='one_generic_param_fn')].inner.generics.params[0].kind.type.synthetic" false
pub fn one_generic_param_fn<T>(_: T) {}

// @has - "$.index[*][?(@.name=='one_synthetic_generic_param_fn')].inner.generics.params[0].kind.type.synthetic" true
pub fn one_synthetic_generic_param_fn(_: impl Clone) {}
