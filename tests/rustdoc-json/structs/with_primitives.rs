// @is "$.index[*][?(@.name=='WithPrimitives')].visibility" \"public\"
// @is "$.index[*][?(@.name=='WithPrimitives')].kind" \"struct\"
// @is "$.index[*][?(@.name=='WithPrimitives')].inner.generics.params[0].name" \"\'a\"
// @is "$.index[*][?(@.name=='WithPrimitives')].inner.generics.params[0].kind.lifetime.outlives" []
// @is "$.index[*][?(@.name=='WithPrimitives')].inner.kind.plain.fields_stripped" true
// @is "$.index[*][?(@.name=='WithPrimitives')].inner.kind.plain.fields" []
pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}
