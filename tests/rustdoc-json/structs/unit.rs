//@ is "$.index[?(@.name=='Unit')].visibility" \"public\"
//@ has "$.index[?(@.name=='Unit')].inner.struct"
//@ is "$.index[?(@.name=='Unit')].inner.struct.kind" \"unit\"
pub struct Unit;
