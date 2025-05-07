//@ count "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.sig.inputs[*]" 1
//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.sig.inputs[0][0]" '"val"'
//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.sig.inputs[0][1].borrowed_ref.lifetime" \"\'c\"
//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.sig.output.primitive" \"i32\"
//@ count "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.generic_params[*]" 1
//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.generic_params[0].name" \"\'c\"
//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type.function_pointer.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
pub type WithHigherRankTraitBounds = for<'c> fn(val: &'c i32) -> i32;
