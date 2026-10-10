//@ has "$.index[?(@.name=='GenericFn')].inner.type_alias"

//@ ismany "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].name" \"\'a\"
//@ has    "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].kind.lifetime"
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.params[*].kind.lifetime.outlives[*]" 0
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.generics.where_predicates[*]" 0
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.type.function_pointer.generic_params[*]" 0
//@ count  "$.index[?(@.name=='GenericFn')].inner.type_alias.type.function_pointer.sig.inputs[*]" 1
//@ is     "$.index[?(@.name=='GenericFn')].inner.type_alias.type.function_pointer.sig.inputs[*][1].borrowed_ref.lifetime" \"\'a\"
//@ is     "$.index[?(@.name=='GenericFn')].inner.type_alias.type.function_pointer.sig.output.borrowed_ref.lifetime" \"\'a\"

pub type GenericFn<'a> = fn(&'a i32) -> &'a i32;

//@ has    "$.index[?(@.name=='ForAll')].inner.type_alias"
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.generics.params[*]" 0
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.generics.where_predicates[*]" 0
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.generic_params[*]" 1
//@ is    "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.generic_params[*].name" \"\'a\"
//@ has   "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.generic_params[*].kind.lifetime"
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.generic_params[*].kind.lifetime.outlives[*]" 0
//@ count "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.sig.inputs[*]" 1
//@ is    "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.sig.inputs[*][1].borrowed_ref.lifetime" \"\'a\"
//@ is    "$.index[?(@.name=='ForAll')].inner.type_alias.type.function_pointer.sig.output.borrowed_ref.lifetime" \"\'a\"
pub type ForAll = for<'a> fn(&'a i32) -> &'a i32;
