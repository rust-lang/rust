// @is "$.index[*][?(@.name=='PlainEmpty')].visibility" \"public\"
// @has "$.index[*][?(@.name=='PlainEmpty')].inner.struct"
// @is "$.index[*][?(@.name=='PlainEmpty')].inner.struct.kind.plain.fields_stripped" false
// @is "$.index[*][?(@.name=='PlainEmpty')].inner.struct.kind.plain.fields" []
pub struct PlainEmpty {}
