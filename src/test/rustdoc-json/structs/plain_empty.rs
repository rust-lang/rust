// @is "$.index[*][?(@.name=='PlainEmpty')].visibility" \"public\"
// @is "$.index[*][?(@.name=='PlainEmpty')].kind" \"struct\"
// @is "$.index[*][?(@.name=='PlainEmpty')].inner.kind.plain.fields_stripped" false
// @is "$.index[*][?(@.name=='PlainEmpty')].inner.kind.plain.fields" []
pub struct PlainEmpty {}
