#![feature(asm)]

#[naked]
//~^ the `#[naked]` attribute is an experimental feature
extern "C" fn naked() {
    asm!("", options(noreturn))
}

#[naked]
//~^ the `#[naked]` attribute is an experimental feature
extern "C" fn naked_2() -> isize {
    asm!("", options(noreturn))
}

fn main() {}
