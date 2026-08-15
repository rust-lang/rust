pub struct Demo(
    i32,
    /// field
    pub i32,
    #[doc(hidden)] i32,
);

//@ set field = "$.index[?(@.docs=='field')].id"

//@ is    "$.index[?(@.name=='Demo')].inner.struct.kind.tuple[0]" null
//@ is    "$.index[?(@.name=='Demo')].inner.struct.kind.tuple[1]" $field
//@ is    "$.index[?(@.name=='Demo')].inner.struct.kind.tuple[2]" null
//@ count "$.index[?(@.name=='Demo')].inner.struct.kind.tuple[*]" 3
