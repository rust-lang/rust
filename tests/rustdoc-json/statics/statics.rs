//@ is '$.index[?(@.name=="A")].inner.static.type' 0
//@ is '$.types[0].primitive' '"i32"'
//@ is '$.index[?(@.name=="A")].inner.static.is_mutable' false
//@ is '$.index[?(@.name=="A")].inner.static.expr' '"5"'
//@ is '$.index[?(@.name=="A")].inner.static.is_unsafe' false
pub static A: i32 = 5;

//@ is '$.index[?(@.name=="B")].inner.static.type' 1
//@ is '$.types[1].primitive' '"u32"'
//@ is '$.index[?(@.name=="B")].inner.static.is_mutable' true
// Expr value isn't gaurenteed, it'd be fine to change it.
//@ is '$.index[?(@.name=="B")].inner.static.expr' '"_"'
//@ is '$.index[?(@.name=="B")].inner.static.is_unsafe' false
pub static mut B: u32 = 2 + 3;
