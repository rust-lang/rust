// skip-filecheck
// EMIT_MIR tls_access.main.PreCodegen.after.mir
//@ compile-flags: -Zmir-opt-level=0

#![feature(thread_local)]

#[thread_local]
static mut FOO: u8 = 3;

fn main() {
    unsafe {
        let a = &FOO;
        FOO = 42;
    }
}
