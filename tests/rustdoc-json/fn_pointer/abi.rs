#![feature(rust_cold_cc)]

//@ jq .index[] | select(.name == "AbiRust").inner.type_alias.type?.function_pointer.header?.abi == "Rust"
pub type AbiRust = fn();

//@ jq .index[] | select(.name == "AbiC").inner.type_alias.type?.function_pointer.header?.abi == {"C": {"unwind": false}}
pub type AbiC = extern "C" fn();

//@ jq .index[] | select(.name == "AbiSystem").inner.type_alias.type?.function_pointer.header?.abi == {"System": {"unwind": false}}
pub type AbiSystem = extern "system" fn();

//@ jq .index[] | select(.name == "AbiCUnwind").inner.type_alias.type?.function_pointer.header?.abi == {"C": {"unwind": true}}
pub type AbiCUnwind = extern "C-unwind" fn();

//@ jq .index[] | select(.name == "AbiSystemUnwind").inner.type_alias.type?.function_pointer.header?.abi == {"System": {"unwind": true}}
pub type AbiSystemUnwind = extern "system-unwind" fn();

//@ jq .index[] | select(.name == "AbiRustCold").inner.type_alias.type?.function_pointer.header?.abi.Other == "\"rust-cold\""
pub type AbiRustCold = extern "rust-cold" fn();
