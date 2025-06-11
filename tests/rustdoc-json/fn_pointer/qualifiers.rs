//@ is "$.index[?(@.name=='FnPointer')].inner.type_alias.type" 0
//@ is "$.types[0].function_pointer.header.is_unsafe" false
//@ is "$.types[0].function_pointer.header.is_const" false
//@ is "$.types[0].function_pointer.header.is_async" false
pub type FnPointer = fn();

//@ is "$.index[?(@.name=='UnsafePointer')].inner.type_alias.type" 1
//@ is "$.types[1].function_pointer.header.is_unsafe" true
//@ is "$.types[1].function_pointer.header.is_const" false
//@ is "$.types[1].function_pointer.header.is_async" false
pub type UnsafePointer = unsafe fn();
