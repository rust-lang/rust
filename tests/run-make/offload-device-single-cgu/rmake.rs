// The device pass bundles each codegen unit's module into `device.bin`, and every CGU writes the
// same file, so a crate split into several CGUs would keep only the kernels of whichever CGU was
// written last. `-Zoffload=Device` therefore forces a single CGU, even over an explicit
// `-Ccodegen-units`. Check that kernels from two modules, which partitioning would otherwise put
// into two CGUs, both end up in the bundle. If we were to enforce fat-lto also for the device, we
// could move the artifact creation to the linker, but enforcing a single CGU is easier for now.
//
// The crate is emitted as an rlib on purpose: `--emit=obj` together with `-o` already resets the
// CGU count to one, which would hide the problem.

//@ needs-offload
//@ needs-llvm-components: amdgpu

use run_make_support::{rfs, rustc, rustc_minicore};

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn main() {
    rustc_minicore()
        .target("amdgcn-amd-amdhsa")
        .target_cpu("gfx90a")
        .output("libminicore.rlib")
        .run();

    rustc()
        .input("device.rs")
        .target("amdgcn-amd-amdhsa")
        .target_cpu("gfx90a")
        .arg("-Zunstable-options")
        .arg("-Zoffload=Device")
        .codegen_units(2)
        .extern_("minicore", "libminicore.rlib")
        .run();

    // The bundle holds the module as bitcode, whose string table keeps symbol names verbatim.
    let device_bin = rfs::read("device.bin");
    for kernel in ["kernel_in_first_module", "kernel_in_second_module"] {
        assert!(contains(&device_bin, kernel), "`{kernel}` is missing from `device.bin`");
    }
}
