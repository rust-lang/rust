// On the amdhsa OS, the host runtime (HIP or HSA) expects a kernel descriptor object for each
// kernel in the ELF file. The amdgpu LLVM backend generates the object. It is created as a symbol
// with the name of the kernel plus a .kd suffix.
// Check that the produced object has the .kd symbol exported.

//@ needs-llvm-components: amdgpu
//@ needs-rust-lld

use run_make_support::{llvm_readobj, rustc};

fn main() {
    rustc()
        .crate_name("foo")
        .target("amdgcn-amd-amdhsa")
        .arg("-Ctarget-cpu=gfx900")
        .crate_type("cdylib")
        .input("foo.rs")
        .run();
    llvm_readobj().input("foo.elf").symbols().run().assert_stdout_contains("kernel.kd");
}
