// Check that, when rustc is invoked with multiple `-Ctarget-cpu` options,
// the last value is used both for the corresponding target modifier in the
// crate metadata and for the target CPU recorded in the LLVM IR.
//@ needs-llvm-components: nvptx

use run_make_support::*;

const TARGET: &str = "nvptx64-nvidia-cuda";
const FIRST_CPU: &str = "sm_70";
const LAST_CPU: &str = "sm_80";

fn main() {
    // Compile lib.rs, emit llvm-ir and metadata
    rustc()
        .input("lib.rs")
        .crate_name("target_cpu_precedence")
        .target(TARGET)
        .target_cpu(FIRST_CPU)
        .target_cpu(LAST_CPU)
        .emit("llvm-ir=output.ll,metadata=output.rmeta")
        .run();

    let llvm_ir = rfs::read_to_string("output.ll");
    // Decode the metadata.
    let target_modifiers =
        rustc().arg("-Zls=target_modifiers").input("output.rmeta").run().stdout_utf8();

    // Make sure the first target cpu did not survive in either artifact.
    assert_not_contains(&llvm_ir, format!(r#""target-cpu"="{FIRST_CPU}""#));
    assert_not_contains(&target_modifiers, format!(r#"target-cpu: Some("{FIRST_CPU}")"#));

    // Combine LLVM-IR and the metadata output into one file.
    // Use FileCheck to verify that both LLVM IR and metadata contain the
    // last mentioned target-cpu
    let filecheck_input = format!(
        "{llvm_ir}\n\
         ; --- target modifiers decoded from output.rmeta ---\n\
         {target_modifiers}"
    );
    llvm_filecheck().patterns("lib.rs").stdin_buf(filecheck_input).run();
}
