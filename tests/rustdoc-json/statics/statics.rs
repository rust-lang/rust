//@ is '$.index[?(@.name=="A")].inner.static.type.primitive' '"i32"'
//@ is '$.index[?(@.name=="A")].inner.static.is_mutable' false
//@ is '$.index[?(@.name=="A")].inner.static.expr' '"5"'
//@ is '$.index[?(@.name=="A")].inner.static.is_unsafe' false
pub static A: i32 = 5;

//@ is '$.index[?(@.name=="B")].inner.static.type.primitive' '"u32"'
//@ is '$.index[?(@.name=="B")].inner.static.is_mutable' true
// Expr value isn't gaurenteed, it'd be fine to change it.
//@ is '$.index[?(@.name=="B")].inner.static.expr' '"_"'
//@ is '$.index[?(@.name=="B")].inner.static.is_unsafe' false
pub static mut B: u32 = 2 + 3;
