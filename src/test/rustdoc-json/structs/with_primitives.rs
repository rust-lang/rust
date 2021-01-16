// @has with_primitives.json "$.index.['0:3'].name" \"WithPrimitives\"
// @has - "$.index.['0:3'].visibility" \"public\"
// @has - "$.index.['0:3'].kind" \"struct\"
// @has - "$.index.['0:3'].inner.generics.params[0].name" \"\'a\"
// @has - "$.index.['0:3'].inner.generics.params[0].kind" \"lifetime\"
// @has - "$.index.['0:3'].inner.struct_type" \"plain\"
// @has - "$.index.['0:3'].inner.fields_stripped" true
pub struct WithPrimitives<'a> {
    num: u32,
    s: &'a str,
}
