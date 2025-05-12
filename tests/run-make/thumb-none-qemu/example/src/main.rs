#![no_main]
#![no_std]
use core::fmt::Write;

use cortex_m::asm;
use cortex_m_rt::entry;
use {cortex_m_semihosting as semihosting, panic_halt as _};

#[entry]
fn main() -> ! {
    let x = 42;

    loop {
        asm::nop();

        // write something through semihosting interface
        let mut hstdout = semihosting::hio::hstdout().unwrap();
        let _ = write!(hstdout, "x = {}\n", x);

        // exit from qemu
        semihosting::debug::exit(semihosting::debug::EXIT_SUCCESS);
    }
}
