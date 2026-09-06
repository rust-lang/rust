// rustc synthesizes some object files with the `object` crate rather than with LLVM (crate
// metadata, and the `symbols.o` it hands to the linker), so their MIPS ELF header flags are
// computed by hand. `EF_MIPS_CPIC` used to be omitted from those whenever the target used the
// static relocation model, but LLVM sets `EF_MIPS_CPIC` on the objects *it* emits unless the
// target turns abicalls off. On a static target that leaves abicalls enabled the two disagree,
// and lld warns "linking abicalls code with non-abicalls code" once per object.
// See <https://github.com/overdrivenpotato/rust-psp/issues/203>.

//@ needs-llvm-components: mips

use run_make_support::{llvm_ar, llvm_readobj, rustc};

fn check(target: &str, expect_cpic: bool) {
    rustc().crate_name("foo").target(target).crate_type("rlib").input("foo.rs").run();

    // `lib.rmeta` is the member rustc builds itself; the rest of the archive comes from LLVM
    // and already carries the right flags, so the archive as a whole cannot be checked.
    llvm_ar().arg("x").arg("libfoo.rlib").arg("lib.rmeta").run();

    // `llvm_readobj()` defaults to the GNU output style, which spells this flag `cpic`. Ask for
    // the LLVM style instead, which names it in full.
    let readobj = llvm_readobj().elf_output_style("LLVM").input("lib.rmeta").file_header().run();
    if expect_cpic {
        readobj.assert_stdout_contains("EF_MIPS_CPIC");
    } else {
        readobj.assert_stdout_not_contains("EF_MIPS_CPIC");
    }
}

fn main() {
    // Static, but abicalls is left enabled, so LLVM marks its objects CPIC and so must we.
    check("mipsel-sony-psp", true);
    check("mipsel-sony-psx", true);

    // Static with `+noabicalls`, so LLVM does not mark its objects CPIC either.
    check("mipsel-unknown-none", false);
}
