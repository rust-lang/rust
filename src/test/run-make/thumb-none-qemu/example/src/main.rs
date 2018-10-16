// #![feature(stdsimd)]
#![no_main]
#![no_std]

extern crate cortex_m;

extern crate cortex_m_rt as rt;
extern crate cortex_m_semihosting as semihosting;
extern crate panic_halt;

use core::fmt::Write;
use cortex_m::asm;
use rt::entry;

entry!(main);

fn main() -> ! {
    let x = 42;

    loop {
        asm::nop();

        // write something through semihosting interface
        let mut hstdout = semihosting::hio::hstdout().unwrap();
        write!(hstdout, "x = {}\n", x);

        // exit from qemu
        semihosting::debug::exit(semihosting::debug::EXIT_SUCCESS);
    }
}
