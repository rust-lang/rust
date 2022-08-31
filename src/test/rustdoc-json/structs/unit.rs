// @has "$.index[*][?(@.name=='Unit')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Unit')].kind" \"struct\"
// @has "$.index[*][?(@.name=='Unit')].inner.struct_type" \"unit\"
// @has "$.index[*][?(@.name=='Unit')].inner.fields" []
pub struct Unit;
