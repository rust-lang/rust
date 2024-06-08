// When rustc received 4 codegen-units, an output path and an emit flag all simultaneously,
// this could cause an annoying recompilation issue, uselessly lengthening the build process.
// A fix was delivered, which resets codegen-units to 1 when necessary,
// but as it directly affected the way codegen-units are manipulated,
// this test was created to check that this fix did not cause compilation failures.
// See https://github.com/rust-lang/rust/issues/30063

//@ ignore-cross-compile

use run_make_support::rustc;
use std::fs;

fn compile(output_file: &str, emit: Option<&str>) {
    let mut rustc = rustc();
    let rustc = rustc.codegen_units(4).output(output_file).input("foo.rs");
    if let Some(emit) = emit {
        rustc.emit(emit);
    }
    rustc.run();
}

fn main() {
    let flags = [
        ("foo-output", None),
        ("asm-output", Some("asm")),
        ("bc-output", Some("llvm-bc")),
        ("ir-output", Some("llvm-ir")),
        ("link-output", Some("link")),
        ("obj-output", Some("obj")),
        ("dep-output", Some("dep-info")),
        ("multi-output", Some("asm,obj")),
    ];
    for (output_file, emit) in flags {
        fs::remove_file(output_file).unwrap_or_default();
        compile(output_file, emit);
        fs::remove_file(output_file);
    }
}
