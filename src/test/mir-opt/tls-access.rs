#![feature(thread_local)]

#[thread_local]
static mut FOO: u8 = 3;

fn main() {
    unsafe {
        let a = &FOO;
        FOO = 42;
    }
}

// EMIT_MIR rustc.main.SimplifyCfg-final.after.mir
