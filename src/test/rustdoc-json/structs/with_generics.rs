use std::collections::HashMap;

// @has "$.index[*][?(@.name=='WithGenerics')].visibility" \"public\"
// @has "$.index[*][?(@.name=='WithGenerics')].kind" \"struct\"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[0].name" \"T\"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[0].kind.type"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[1].name" \"U\"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[1].kind.type"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.struct_type" \"plain\"
// @has "$.index[*][?(@.name=='WithGenerics')].inner.fields_stripped" true
pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
