use std::collections::HashMap;

// @has with_generics.json "$.index.['0:4'].name" \"WithGenerics\"
// @has - "$.index.['0:4'].visibility" \"public\"
// @has - "$.index.['0:4'].kind" \"struct\"
// @has - "$.index.['0:4'].inner.generics.params[0].name" \"T\"
// @has - "$.index.['0:4'].inner.generics.params[0].kind.type"
// @has - "$.index.['0:4'].inner.generics.params[1].name" \"U\"
// @has - "$.index.['0:4'].inner.generics.params[1].kind.type"
// @has - "$.index.['0:4'].inner.struct_type" \"plain\"
// @has - "$.index.['0:4'].inner.fields_stripped" true
pub struct WithGenerics<T, U> {
    stuff: Vec<T>,
    things: HashMap<U, U>,
}
