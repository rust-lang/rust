use std::collections::HashMap;

//@ is "$.index[?(@.name=='WithGenerics')].visibility" \"public\"
//@ has "$.index[?(@.name=='WithGenerics')].inner.struct"
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.generics.params[0].name" \"T\"
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.generics.params[0].kind.type.bounds" []
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.generics.params[1].name" \"U\"
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.generics.params[1].kind.type.bounds" []
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.kind.plain.has_stripped_fields" true
//@ is "$.index[?(@.name=='WithGenerics')].inner.struct.kind.plain.fields" []
pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
