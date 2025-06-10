//@ has "$.index[?(@.name=='Union')].visibility" \"public\"
//@ has "$.index[?(@.name=='Union')].inner.union"
//@ !has "$.index[?(@.name=='Union')].inner.union.kind"
//@ set Union = "$.index[?(@.name=='Union')].id"
pub union Union {
    int: i32,
    float: f32,
}

//@ is "$.index[?(@.name=='make_int_union')].inner.function.sig.output" 0
//@ has "$.types[0].resolved_path"
//@ is "$.types[0].resolved_path.id" $Union
pub fn make_int_union(int: i32) -> Union {
    Union { int }
}
