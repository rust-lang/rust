//@ is "$.index[?(@.name=='WithHigherRankTraitBounds')].inner.type_alias.type" 2
//@ count "$.types[2].function_pointer.sig.inputs[*]" 1
//@ is "$.types[2].function_pointer.sig.inputs[0][0]" '"val"'
//@ is "$.types[2].function_pointer.sig.inputs[0][1]" 1
//@ is "$.types[1].borrowed_ref.lifetime" \"\'c\"
//@ is "$.types[2].function_pointer.sig.output" 0
//@ is "$.types[0].primitive" \"i32\"
//@ count "$.types[2].function_pointer.generic_params[*]" 1
//@ is "$.types[2].function_pointer.generic_params[0].name" \"\'c\"
//@ is "$.types[2].function_pointer.generic_params[0].kind" '{ "lifetime": { "outlives": [] } }'
pub type WithHigherRankTraitBounds = for<'c> fn(val: &'c i32) -> i32;
