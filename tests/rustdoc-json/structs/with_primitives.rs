//@ is "$.index[?(@.name=='WithPrimitives')].visibility" \"public\"
//@ has "$.index[?(@.name=='WithPrimitives')].inner.struct"
//@ is "$.index[?(@.name=='WithPrimitives')].inner.struct.generics.params[0].name" \"\'a\"
//@ is "$.index[?(@.name=='WithPrimitives')].inner.struct.generics.params[0].kind.lifetime.outlives" []
//@ is "$.index[?(@.name=='WithPrimitives')].inner.struct.kind.plain.has_stripped_fields" true
//@ is "$.index[?(@.name=='WithPrimitives')].inner.struct.kind.plain.fields" []
pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}
