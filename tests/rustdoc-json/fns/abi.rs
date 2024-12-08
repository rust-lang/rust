// ignore-tidy-linelength

#![feature(abi_vectorcall)]

//@ is "$.index[*][?(@.name=='abi_rust')].inner.function.header.abi" \"Rust\"
pub fn abi_rust() {}

//@ is "$.index[*][?(@.name=='abi_c')].inner.function.header.abi" '{"C": {"unwind": false}}'
pub extern "C" fn abi_c() {}

//@ is "$.index[*][?(@.name=='abi_system')].inner.function.header.abi" '{"System": {"unwind": false}}'
pub extern "system" fn abi_system() {}

//@ is "$.index[*][?(@.name=='abi_c_unwind')].inner.function.header.abi" '{"C": {"unwind": true}}'
pub extern "C-unwind" fn abi_c_unwind() {}

//@ is "$.index[*][?(@.name=='abi_system_unwind')].inner.function.header.abi" '{"System": {"unwind": true}}'
pub extern "system-unwind" fn abi_system_unwind() {}

//@ is "$.index[*][?(@.name=='abi_vectorcall')].inner.function.header.abi.Other" '"\"vectorcall\""'
pub extern "vectorcall" fn abi_vectorcall() {}

//@ is "$.index[*][?(@.name=='abi_vectorcall_unwind')].inner.function.header.abi.Other" '"\"vectorcall-unwind\""'
pub extern "vectorcall-unwind" fn abi_vectorcall_unwind() {}
