use core::arch::global_asm;

global_asm!(include_str!("v810/memcpy_wordaligned.s"), options(raw));
