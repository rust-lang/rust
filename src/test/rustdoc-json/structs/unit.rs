// @is "$.index[*][?(@.name=='Unit')].visibility" \"public\"
// @is "$.index[*][?(@.name=='Unit')].kind" \"struct\"
// @is "$.index[*][?(@.name=='Unit')].inner.kind" \"unit\"
pub struct Unit;
