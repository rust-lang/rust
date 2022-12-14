// @has "$.index[*][?(@.name=='Union')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Union')].kind" \"union\"
// @!has "$.index[*][?(@.name=='Union')].inner.struct_type"
// @set Union = "$.index[*][?(@.name=='Union')].id"
pub union Union {
    int: i32,
    float: f32,
}


// @is "$.index[*][?(@.name=='make_int_union')].inner.decl.output.kind" '"resolved_path"'
// @is "$.index[*][?(@.name=='make_int_union')].inner.decl.output.inner.id" $Union
pub fn make_int_union(int: i32) -> Union {
    Union { int }
}
