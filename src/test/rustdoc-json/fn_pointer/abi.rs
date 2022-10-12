// ignore-tidy-linelength

#![feature(abi_vectorcall)]
#![feature(c_unwind)]

// @is "$.index[*][?(@.name=='AbiRust')].inner.type.inner.header.abi" \"Rust\"
pub type AbiRust = fn();

// @is "$.index[*][?(@.name=='AbiC')].inner.type.inner.header.abi" '{"C": {"unwind": false}}'
pub type AbiC = extern "C" fn();

// @is "$.index[*][?(@.name=='AbiSystem')].inner.type.inner.header.abi" '{"System": {"unwind": false}}'
pub type AbiSystem = extern "system" fn();

// @is "$.index[*][?(@.name=='AbiCUnwind')].inner.type.inner.header.abi" '{"C": {"unwind": true}}'
pub type AbiCUnwind = extern "C-unwind" fn();

// @is "$.index[*][?(@.name=='AbiSystemUnwind')].inner.type.inner.header.abi" '{"System": {"unwind": true}}'
pub type AbiSystemUnwind = extern "system-unwind" fn();

// @is "$.index[*][?(@.name=='AbiVecorcall')].inner.type.inner.header.abi.Other" '"\"vectorcall\""'
pub type AbiVecorcall = extern "vectorcall" fn();

// @is "$.index[*][?(@.name=='AbiVecorcallUnwind')].inner.type.inner.header.abi.Other" '"\"vectorcall-unwind\""'
pub type AbiVecorcallUnwind = extern "vectorcall-unwind" fn();
