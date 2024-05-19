// ignore-tidy-linelength

// @is "$.index[*][?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.unsafe" false
// @is "$.index[*][?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.const" false
// @is "$.index[*][?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.async" false
pub type FnPointer = fn();

// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.unsafe" true
// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.const" false
// @is "$.index[*][?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.async" false
pub type UnsafePointer = unsafe fn();
