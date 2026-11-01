//@ build-fail
//@ only-x86_64-unknown-linux-gnu
//@ dont-check-compiler-stderr
//@ dont-check-compiler-stdout
//@ ignore-backends: gcc

// https://github.com/rust-lang/rust/issues/151950

unsafe extern "C" {
    #[link_name = "memset]; mov eax, 1; #"]
    unsafe fn inject();
}

#[unsafe(export_name = "memset]; mov eax, 1; #")]
extern "C" fn inject_() {}

#[unsafe(naked)]
extern "C" fn print_0() -> usize {
    core::arch::naked_asm!("lea rax, [{}]", "ret", sym inject)
}

#[unsafe(naked)]
extern "C" fn print_1() -> usize {
    core::arch::naked_asm!("lea rax, [{}]", "ret", sym inject_)
}

fn main() {
    dbg!(print_0());
    dbg!(print_1());
}

//~? ERROR linking
