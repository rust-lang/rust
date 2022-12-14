// @is "$.index[*][?(@.name=='FnPointer')].inner.type.inner.header.unsafe" false
// @is "$.index[*][?(@.name=='FnPointer')].inner.type.inner.header.const" false
// @is "$.index[*][?(@.name=='FnPointer')].inner.type.inner.header.async" false
pub type FnPointer = fn();

// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type.inner.header.unsafe" true
// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type.inner.header.const" false
// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type.inner.header.async" false
pub type UnsafePointer = unsafe fn();
