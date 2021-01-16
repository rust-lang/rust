// @has unit.json "$.index.['0:3'].name" \"Unit\"
// @has - "$.index.['0:3'].visibility" \"public\"
// @has - "$.index.['0:3'].kind" \"struct\"
// @has - "$.index.['0:3'].inner.struct_type" \"unit\"
// @has - "$.index.['0:3'].inner.fields" []
pub struct Unit;
