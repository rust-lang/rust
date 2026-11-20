// Compiler:

// Regression test for https://github.com/rust-lang/rustc_codegen_gcc/issues/827

#![crate_type = "lib"]

#[cfg(target_arch = "x86_64")]
pub type NoReturn = extern "sysv64" fn(&'static u8) -> !;

#[cfg(target_arch = "x86_64")]
pub fn call_no_return(function: *const NoReturn) -> ! {
    unsafe {
        std::arch::asm!("call {}", in(reg) function, options(noreturn));
    }
}
