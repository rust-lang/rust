// @is "$.index[*][?(@.name=='Tuple')].visibility" \"public\"
// @is "$.index[*][?(@.name=='Tuple')].kind" \"struct\"
// @is "$.index[*][?(@.name=='Tuple')].inner.kind.tuple" '[null, null]'
pub struct Tuple(u32, String);
