use std::collections::HashMap;

// @is "$.index[*][?(@.name=='WithGenerics')].visibility" \"public\"
// @is "$.index[*][?(@.name=='WithGenerics')].kind" \"struct\"
// @is "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[0].name" \"T\"
// @is "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[0].kind.type.bounds" []
// @is "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[1].name" \"U\"
// @is "$.index[*][?(@.name=='WithGenerics')].inner.generics.params[1].kind.type.bounds" []
// @is "$.index[*][?(@.name=='WithGenerics')].inner.kind.plain.fields_stripped" true
// @is "$.index[*][?(@.name=='WithGenerics')].inner.kind.plain.fields" []
pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
