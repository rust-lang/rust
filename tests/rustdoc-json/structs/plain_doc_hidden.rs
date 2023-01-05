pub struct Demo {
    pub x: i32,
    #[doc(hidden)]
    pub y: i32,
}

// @set x = "$.index[*][?(@.name=='x')].id"
// @!has "$.index[*][?(@.name=='y')].id"
// @is "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields[0]" $x
// @count "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields[*]" 1
// @is "$.index[*][?(@.name=='Demo')].inner.kind.plain.fields_stripped" true
