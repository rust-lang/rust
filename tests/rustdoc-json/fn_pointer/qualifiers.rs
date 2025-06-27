//@ jq .index[] | select(.name == "FnPointer").inner.type_alias.type?.function_pointer.header? | [.is_unsafe, .is_const, .is_async] == [false, false, false]
pub type FnPointer = fn();

//@ jq .index[] | select(.name == "UnsafePointer").inner.type_alias.type?.function_pointer.header? | [.is_unsafe, .is_const, .is_async] == [true, false, false]
pub type UnsafePointer = unsafe fn();
