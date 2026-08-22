// Verifies that #[link_section] works on foreign (extern) items.
// This is only supported on BPF targets.
//
//@ only-bpf
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none -C no-prepopulate-passes

#![no_std]
#![no_main]
#![crate_type = "lib"]

extern "C" {
    // CHECK: @EXTERN_STATIC = external global i32, section ".ksyms"
    #[link_section = ".ksyms"]
    pub static EXTERN_STATIC: i32;
}

extern "C" {
    // CHECK: declare {{.*}}void @extern_fn(){{.*}} section ".ksyms"
    #[link_section = ".ksyms"]
    pub fn extern_fn();
}

#[no_mangle]
pub fn use_extern_items() -> i32 {
    unsafe {
        extern_fn();
        EXTERN_STATIC
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
