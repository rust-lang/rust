// ignore-tidy-linelength

// @is "$.index[*][?(@.name=='GenericFn')].kind" \"typedef\"

// @ismany "$.index[*][?(@.name=='GenericFn')].inner.generics.params[*].name" \"\'a\"
// @has    "$.index[*][?(@.name=='GenericFn')].inner.generics.params[*].kind.lifetime"
// @count  "$.index[*][?(@.name=='GenericFn')].inner.generics.params[*].kind.lifetime.outlives[*]" 0
// @count  "$.index[*][?(@.name=='GenericFn')].inner.generics.where_predicates[*]" 0
// @is     "$.index[*][?(@.name=='GenericFn')].inner.type.kind" \"function_pointer\"
// @count  "$.index[*][?(@.name=='GenericFn')].inner.type.inner.generic_params[*]" 0
// @count  "$.index[*][?(@.name=='GenericFn')].inner.type.inner.decl.inputs[*]" 1
// @is     "$.index[*][?(@.name=='GenericFn')].inner.type.inner.decl.inputs[*][1].inner.lifetime" \"\'a\"
// @is     "$.index[*][?(@.name=='GenericFn')].inner.type.inner.decl.output.inner.lifetime" \"\'a\"

pub type GenericFn<'a> = fn(&'a i32) -> &'a i32;

// @is    "$.index[*][?(@.name=='ForAll')].kind" \"typedef\"
// @count "$.index[*][?(@.name=='ForAll')].inner.generics.params[*]" 0
// @count "$.index[*][?(@.name=='ForAll')].inner.generics.where_predicates[*]" 0
// @count "$.index[*][?(@.name=='ForAll')].inner.type.inner.generic_params[*]" 1
// @is    "$.index[*][?(@.name=='ForAll')].inner.type.inner.generic_params[*].name" \"\'a\"
// @has   "$.index[*][?(@.name=='ForAll')].inner.type.inner.generic_params[*].kind.lifetime"
// @count "$.index[*][?(@.name=='ForAll')].inner.type.inner.generic_params[*].kind.lifetime.outlives[*]" 0
// @count "$.index[*][?(@.name=='ForAll')].inner.type.inner.decl.inputs[*]" 1
// @is    "$.index[*][?(@.name=='ForAll')].inner.type.inner.decl.inputs[*][1].inner.lifetime" \"\'a\"
// @is    "$.index[*][?(@.name=='ForAll')].inner.type.inner.decl.output.inner.lifetime" \"\'a\"
pub type ForAll = for<'a> fn(&'a i32) -> &'a i32;
