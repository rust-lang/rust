// A leading `\x01` suppresses decoration; it is not part of the exported name.

//@ only-elf
//@ ignore-cross-compile
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

use run_make_support::{dynamic_lib_name, llvm_readobj, rfs, rust_lib_name, rustc};

fn main() {
    rustc().input("up.rs").crate_type("rlib").run();
    rustc().input("foo.rs").crate_type("cdylib").extern_("up", rust_lib_name("up")).run();

    llvm_readobj()
        .arg("--dyn-syms")
        .input(dynamic_lib_name("foo"))
        .run()
        .assert_stdout_contains(" ctl_export\n")
        .assert_stdout_contains(" plain_export\n")
        // Exported by the upstream rlib only, so its symbol comes from `symbols.o`.
        .assert_stdout_contains(" up_export\n");

    rfs::remove_file(rust_lib_name("up"));
}
