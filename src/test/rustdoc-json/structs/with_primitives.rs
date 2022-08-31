// @has "$.index[*][?(@.name=='WithPrimitives')].visibility" \"public\"
// @has "$.index[*][?(@.name=='WithPrimitives')].kind" \"struct\"
// @has "$.index[*][?(@.name=='WithPrimitives')].inner.generics.params[0].name" \"\'a\"
// @has "$.index[*][?(@.name=='WithPrimitives')].inner.generics.params[0].kind.lifetime.outlives" []
// @has "$.index[*][?(@.name=='WithPrimitives')].inner.struct_type" \"plain\"
// @has "$.index[*][?(@.name=='WithPrimitives')].inner.fields_stripped" true
pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}
