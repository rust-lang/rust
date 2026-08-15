#![feature(rust_cold_cc)]

//@ is "$.index[?(@.name=='AbiRust')].inner.type_alias.type.function_pointer.header.abi" \"Rust\"
pub type AbiRust = fn();

//@ is "$.index[?(@.name=='AbiC')].inner.type_alias.type.function_pointer.header.abi" '{"C": {"unwind": false}}'
pub type AbiC = extern "C" fn();

//@ is "$.index[?(@.name=='AbiSystem')].inner.type_alias.type.function_pointer.header.abi" '{"System": {"unwind": false}}'
pub type AbiSystem = extern "system" fn();

//@ is "$.index[?(@.name=='AbiCUnwind')].inner.type_alias.type.function_pointer.header.abi" '{"C": {"unwind": true}}'
pub type AbiCUnwind = extern "C-unwind" fn();

//@ is "$.index[?(@.name=='AbiSystemUnwind')].inner.type_alias.type.function_pointer.header.abi" '{"System": {"unwind": true}}'
pub type AbiSystemUnwind = extern "system-unwind" fn();

//@ is "$.index[?(@.name=='AbiRustCold')].inner.type_alias.type.function_pointer.header.abi.Other" '"\"rust-cold\""'
pub type AbiRustCold = extern "rust-cold" fn();
