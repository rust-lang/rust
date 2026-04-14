#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::device::HostState;
use crate::syscalls;
use anyhow::Result;
use wasmi::Linker;

pub fn register_imports(linker: &mut Linker<HostState>) -> Result<()> {
    linker.func_wrap("thing.sys", "log", syscalls::log)?;
    linker.func_wrap("thing.sys", "mmio_read32", syscalls::mmio_read32)?;
    linker.func_wrap("thing.sys", "mmio_write32", syscalls::mmio_write32)?;
    linker.func_wrap("thing.sys", "sleep_ms", syscalls::sleep_ms)?;
    Ok(())
}
