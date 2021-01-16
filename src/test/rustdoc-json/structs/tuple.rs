// @has tuple.json "$.index.['0:3'].name" \"Tuple\"
// @has - "$.index.['0:3'].visibility" \"public\"
// @has - "$.index.['0:3'].kind" \"struct\"
// @has - "$.index.['0:3'].inner.struct_type" \"tuple\"
// @has - "$.index.['0:3'].inner.fields_stripped" true
pub struct Tuple(u32, String);
