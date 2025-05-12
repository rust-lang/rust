// test that "error: arguments for inline assembly must be copyable" doesn't show up in this code
//@ needs-asm-support
//@ only-x86_64
fn main() {
    let peb: *const PEB; //~ ERROR cannot find type `PEB` in this scope [E0412]
    unsafe { std::arch::asm!("mov {0}, fs:[0x30]", out(reg) peb); }
}
