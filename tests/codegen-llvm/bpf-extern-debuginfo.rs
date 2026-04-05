// Verifies that BPF extern declarations get correct debug info for BTF generation.
// Extern statics should have isDefinition: false in DIGlobalVariable.
// Extern functions should NOT have SPFlagDefinition in DISubprogram.
//
//@ only-bpf
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none -C debuginfo=2

#![no_std]
#![no_main]
#![crate_type = "lib"]

extern "C" {
    // CHECK: !DIGlobalVariable(name: "KERNEL_VERSION"
    // CHECK-SAME: isLocal: false
    // CHECK-SAME: isDefinition: false
    #[link_section = ".ksyms"]
    pub static KERNEL_VERSION: u64;
}

extern "C" {
    // CHECK: !DISubprogram(name: "bpf_kfunc"
    // CHECK-NOT: DISPFlagDefinition
    // CHECK-SAME: templateParams:
    #[link_section = ".ksyms"]
    pub fn bpf_kfunc(x: u64) -> u64;
}

#[no_mangle]
pub fn test_extern_items() -> u64 {
    unsafe { KERNEL_VERSION + bpf_kfunc(42) }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
