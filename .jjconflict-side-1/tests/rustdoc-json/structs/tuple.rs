//@ is "$.index[?(@.name=='Tuple')].visibility" \"public\"
//@ has "$.index[?(@.name=='Tuple')].inner.struct"
//@ is "$.index[?(@.name=='Tuple')].inner.struct.kind.tuple" '[null, null]'
pub struct Tuple(u32, String);
