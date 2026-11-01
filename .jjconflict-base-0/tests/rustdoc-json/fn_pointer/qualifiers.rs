//@ is "$.index[?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.is_unsafe" false
//@ is "$.index[?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.is_const" false
//@ is "$.index[?(@.name=='FnPointer')].inner.type_alias.type.function_pointer.header.is_async" false
pub type FnPointer = fn();

//@ is "$.index[?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.is_unsafe" true
//@ is "$.index[?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.is_const" false
//@ is "$.index[?(@.name=='UnsafePointer')].inner.type_alias.type.function_pointer.header.is_async" false
pub type UnsafePointer = unsafe fn();
