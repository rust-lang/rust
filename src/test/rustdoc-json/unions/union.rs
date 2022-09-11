// @has "$.index[*][?(@.name=='Union')].visibility" \"public\"
// @has "$.index[*][?(@.name=='Union')].kind" \"union\"
// @!has "$.index[*][?(@.name=='Union')].inner.struct_type"
pub union Union {
    int: i32,
    float: f32,
}
