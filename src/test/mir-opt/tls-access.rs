#![feature(thread_local)]
#![feature(bench_black_box)]

#[thread_local]
static mut FOO: u8 = 3;

fn main() {
    unsafe {
        let a = &FOO;
        FOO = 42;
        core::hint::black_box(());
    }
}

// EMIT_MIR tls_access.main.SimplifyCfg-final.after.mir
