// @has header.json "$.index[*][?(@.name=='FnPointer')].inner.type.inner.header" "[]"
pub type FnPointer = fn();

// @has - "$.index[*][?(@.name=='UnsafePointer')].inner.type.inner.header" '["unsafe"]'
pub type UnsafePointer = unsafe fn();
