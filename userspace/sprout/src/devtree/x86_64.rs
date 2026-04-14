#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use super::DevTreeCtx;
use abi::schema::{confidence, keys, kinds, rels, source};
use stem::info;
use stem::thing::sys as thingsys;

#[allow(dead_code)]
pub fn enumerate(_ctx: &DevTreeCtx) -> Result<(), ()> {
    info!("SPROUT: x86_64 platform enrichment disabled (migrating to VFS)");
    Ok(())
}
