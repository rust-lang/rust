//@ ignore-auxiliary (used by `./inner-cfg-non-inline-mod.rs`)

#![cfg_attr(true, cfg(false))]
