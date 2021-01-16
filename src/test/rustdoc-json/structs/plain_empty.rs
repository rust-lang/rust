// @has plain_empty.json "$.index.['0:3'].name" \"PlainEmpty\"
// @has - "$.index.['0:3'].visibility" \"public\"
// @has - "$.index.['0:3'].kind" \"struct\"
// @has - "$.index.['0:3'].inner.struct_type" \"plain\"
// @has - "$.index.['0:3'].inner.fields_stripped" false
// @has - "$.index.['0:3'].inner.fields" []
pub struct PlainEmpty {}
