#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use super::DevTreeCtx;
use stem::info;

#[allow(dead_code)]
pub fn enumerate(_ctx: &DevTreeCtx) -> Result<(), ()> {
    info!("SPROUT: Enumerating AArch64 platform (stub)...");
    Ok(())
}
