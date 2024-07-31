// If `-o -` or `--emit KIND=-` is provided, output should be written
// to stdout instead. Binary output (`obj`, `llvm-bc`, `link` and
// `metadata`) being written this way will result in an error unless
// stdout is not a tty. Multiple output types going to stdout will
// trigger an error too, as they will all be mixed together.
// See https://github.com/rust-lang/rust/pull/111626

use run_make_support::{diff, rfs, rustc};

fn main() {
    rfs::create_dir("out");
    let tests = ["asm", "llvm-ir", "dep-info", "mir", "llvm-bc", "obj", "metadata", "link"];
    for test in tests {
        test_emit(test);
    }
    // These two last tests, which combine multiple emit types, should be done separately.
    diff()
        .expected_file("emit-multiple-types.stderr")
        .actual_text(
            "actual",
            rustc()
                .output("-")
                .emit("asm=-")
                .emit("llvm-ir=-")
                .emit("dep-info=-")
                .emit("mir=-")
                .input("test.rs")
                .run_fail()
                .stderr_utf8(),
        )
        .run();
    diff()
        .expected_file("emit-multiple-types.stderr")
        .actual_text(
            "actual",
            rustc()
                .output("-")
                .emit("asm,llvm-ir,dep-info,mir")
                .input("test.rs")
                .run_fail()
                .stderr_utf8(),
        )
        .run();
}

fn test_emit(emit_type: &str) {
    // Emitting these types will cause a compilation failure, which should be compared to a
    // blessed stderr file for differences.
    let stderr_types = ["llvm-bc", "obj", "metadata", "link"];
    // Successful types (not in stderr_types) should start by outputting one emit file.
    if !stderr_types.contains(&emit_type) {
        let mut initial_compile = rustc();
        initial_compile.emit(&format!("{emit_type}=out/{emit_type}")).input("test.rs");
        // dep-info requires an extra unstable argument.
        if emit_type == "dep-info" {
            initial_compile.arg("-Zdep-info-omit-d-target=yes");
        }
        initial_compile.run();
    }
    let mut compile = rustc();
    compile.emit(&format!("{emit_type}=-")).input("test.rs");
    // Check if compilation should succeed or fail depending on the emit type.
    let compile =
        if stderr_types.contains(&emit_type) { compile.run_fail() } else { compile.run() };
    let emit = if stderr_types.contains(&emit_type) {
        compile.stderr_utf8()
    } else {
        compile.stdout_utf8()
    };
    let mut diff = diff();
    // Compare the output with either an emit file or stderr file, depending on success
    // or failure.
    if stderr_types.contains(&emit_type) {
        diff.expected_file(&format!("emit-{emit_type}.stderr"));
    } else {
        diff.expected_file(&format!("out/{emit_type}"));
    }
    diff.actual_text("actual", &emit).run();
}
