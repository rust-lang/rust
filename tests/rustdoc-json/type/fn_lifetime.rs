//@ has "$.index[?(@.name=='GenericFn')].inner.type_alias"

//@ ismany "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].name" \"\'a\"
//@ has    "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].kind.lifetime"
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].kind.lifetime.outlives[*]" 0
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.where_predicates[*]" 0
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.type.function_pointer.generic_params[*]" 0
//@ is     "$.index[?(@.name=='GenericFn')].inner.type_alias.type" 2
//@ count  "$.types[2].function_pointer.sig.inputs[*]" 1
//@ is     "$.types[2].function_pointer.sig.inputs[*][1]" 1
//@ is     "$.types[1].borrowed_ref.lifetime" \"\'a\"
//@ is     "$.types[2].function_pointer.sig.output" 1

pub type GenericFn<'a> = fn(&'a i32) -> &'a i32;

//@ has    "$.index[?(@.name=='ForAll')].inner.type_alias"
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.generics.params[*]" 0
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.generics.where_predicates[*]" 0
//@ is    "$.index[?(@.name=='ForAll')].inner.type_alias.type" 3
//@ count "$.types[3].function_pointer.generic_params[*]" 1
//@ is    "$.types[3].function_pointer.generic_params[*].name" \"\'a\"
//@ has   "$.types[3].function_pointer.generic_params[*].kind.lifetime"
//@ count "$.types[3].function_pointer.generic_params[*].kind.lifetime.outlives[*]" 0
//@ count "$.types[3].function_pointer.sig.inputs[*]" 1
//@ is    "$.types[3].function_pointer.sig.inputs[*][1]" 1
//@ is    "$.types[3].function_pointer.sig.output" 1
pub type ForAll = for<'a> fn(&'a i32) -> &'a i32;
