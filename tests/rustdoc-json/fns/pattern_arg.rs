// @is "$.index[*][?(@.name=='fst')].inner.function.decl.inputs[0][0]" '"(x, _)"'
pub fn fst<X, Y>((x, _): (X, Y)) -> X {
    x
}

// @is "$.index[*][?(@.name=='drop_int')].inner.function.decl.inputs[0][0]" '"_"'
pub fn drop_int(_: i32) {}
