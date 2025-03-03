#![feature(abi_vectorcall)]

//@ is "$.index[*][?(@.name=='AbiRust')].inner.type_alias.type.function_pointer.header.abi" \"Rust\"
pub type AbiRust = fn();

//@ is "$.index[*][?(@.name=='AbiC')].inner.type_alias.type.function_pointer.header.abi" '{"C": {"unwind": false}}'
pub type AbiC = extern "C" fn();

//@ is "$.index[*][?(@.name=='AbiSystem')].inner.type_alias.type.function_pointer.header.abi" '{"System": {"unwind": false}}'
pub type AbiSystem = extern "system" fn();

//@ is "$.index[*][?(@.name=='AbiCUnwind')].inner.type_alias.type.function_pointer.header.abi" '{"C": {"unwind": true}}'
pub type AbiCUnwind = extern "C-unwind" fn();

//@ is "$.index[*][?(@.name=='AbiSystemUnwind')].inner.type_alias.type.function_pointer.header.abi" '{"System": {"unwind": true}}'
pub type AbiSystemUnwind = extern "system-unwind" fn();

//@ is "$.index[*][?(@.name=='AbiVecorcall')].inner.type_alias.type.function_pointer.header.abi.Other" '"\"vectorcall\""'
pub type AbiVecorcall = extern "vectorcall" fn();

//@ is "$.index[*][?(@.name=='AbiVecorcallUnwind')].inner.type_alias.type.function_pointer.header.abi.Other" '"\"vectorcall-unwind\""'
pub type AbiVecorcallUnwind = extern "vectorcall-unwind" fn();
