#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::{error, info};

mod devtree;
mod ledger;
mod pipelines;
mod registry;
mod supervisor;
mod task;

#[stem::main]
fn main(arg0: usize) -> ! {
    let cpu = stem::arch::whoami();
    stem::debug!(
        "[sprout] whoami: cs=0x{:x} ss=0x{:x} cpl={} rsp=0x{:x} rip=0x{:x} rflags=0x{:x}",
        cpu.cs,
        cpu.ss,
        cpu.cpl,
        cpu.rsp,
        cpu.rip,
        cpu.rflags
    );

    info!("SPROUT: v0.4.1 [REBUILT] starting (Supervisor Mode)...");

    match devtree::init() {
        Ok(ctx) => {
            if let Err(_) = devtree::build(&ctx) {
                error!("SPROUT: Failed to build device tree!");
            }
        }
        Err(_) => {
            error!("SPROUT: Failed to initialize devtree context! (continuing)");
        }
    }

    stem::debug!("SPROUT: About to create Supervisor...");
    stem::debug!("SPROUT: Listing /bin directory...");
    if let Ok(fd) = stem::syscall::vfs::vfs_open("/bin", stem::abi::syscall::vfs_flags::O_RDONLY) {
        let mut buf = [0u8; 4096];
        if let Ok(n) = stem::syscall::vfs::vfs_read(fd, &mut buf) {
            stem::debug!("SPROUT: /bin dir content (raw): {:?}", &buf[..n]);
        }
        let _ = stem::syscall::vfs::vfs_close(fd);
    }
    let mut sup = supervisor::Supervisor::new(arg0);
    stem::debug!("SPROUT: Supervisor created, calling run_forever...");
    sup.run_forever()
}
