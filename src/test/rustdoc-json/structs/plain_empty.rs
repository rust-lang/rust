// @has "$.index[*][?(@.name=='PlainEmpty')].visibility" \"public\"
// @has "$.index[*][?(@.name=='PlainEmpty')].kind" \"struct\"
// @has "$.index[*][?(@.name=='PlainEmpty')].inner.struct_type" \"plain\"
// @has "$.index[*][?(@.name=='PlainEmpty')].inner.fields_stripped" false
// @has "$.index[*][?(@.name=='PlainEmpty')].inner.fields" []
pub struct PlainEmpty {}
