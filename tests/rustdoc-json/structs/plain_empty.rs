//@ is "$.index[?(@.name=='PlainEmpty')].visibility" \"public\"
//@ has "$.index[?(@.name=='PlainEmpty')].inner.struct"
//@ is "$.index[?(@.name=='PlainEmpty')].inner.struct.kind.plain.has_stripped_fields" false
//@ is "$.index[?(@.name=='PlainEmpty')].inner.struct.kind.plain.fields" []
pub struct PlainEmpty {}
