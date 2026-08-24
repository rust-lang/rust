use run_make_support::{llvm_bcanalyzer, rustc};

fn main() {
    rustc().input("foo.rs").crate_type("lib").arg("-Clto=fat").arg("--emit=llvm-bc").run();

    llvm_bcanalyzer()
        .input("foo.bc")
        .run()
        .assert_stdout_contains("FULL_LTO_GLOBALVAL_SUMMARY_BLOCK");
}
