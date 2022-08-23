// @has "$.index[*][?(@.name=='Tuple')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Tuple')].kind" \"struct\"
// @has "$.index[*][?(@.name=='Tuple')].inner.struct_type" \"tuple\"
// @has "$.index[*][?(@.name=='Tuple')].inner.fields_stripped" true
pub struct Tuple(u32, String);
