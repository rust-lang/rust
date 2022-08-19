// @has "$.index[*][?(@.name=='Unit')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Unit')].kind" \"struct\"
// @has "$.index[*][?(@.name=='Unit')].inner.kind" \"unit\"
// @has "$.index[*][?(@.name=='Unit')].inner.fields" []
pub struct Unit;
