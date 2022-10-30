pub struct Demo {
    pub x: i32,
    pub y: i32,
}

// @set x = "$.index[*][?(@.name=='x')].id"
// @set y = "$.index[*][?(@.name=='y')].id"
// @is "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields[0]" $x
// @is "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields[1]" $y
// @count "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields[*]" 2
// @is "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields_stripped" false
