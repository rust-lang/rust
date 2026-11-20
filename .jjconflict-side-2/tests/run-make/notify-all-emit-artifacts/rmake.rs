//@ needs-target-std
//
// rust should produce artifact notifications about files it was asked to --emit.
//
// It should work in incremental mode both on the first pass where files are generated as well
// as on subsequent passes where they are taken from the incremental cache
//
// See <https://internals.rust-lang.org/t/easier-access-to-files-generated-by-emit-foo/20477>
extern crate run_make_support;

use run_make_support::{cwd, rustc};

fn main() {
    // With single codegen unit files are renamed to match the source file name
    for _ in 0..=1 {
        let output = rustc()
            .input("lib.rs")
            .emit("obj,asm,llvm-ir,llvm-bc,mir")
            .codegen_units(1)
            .json("artifacts")
            .error_format("json")
            .incremental(cwd())
            .run();
        let stderr = output.stderr_utf8();
        for file in &["lib.o", "lib.ll", "lib.bc", "lib.s"] {
            assert!(stderr.contains(file), "No {:?} in {:?}", file, stderr);
        }
    }

    // with multiple codegen units files keep codegen unit id part.
    for _ in 0..=1 {
        let output = rustc()
            .input("lib.rs")
            .emit("obj,asm,llvm-ir,llvm-bc,mir")
            .codegen_units(2)
            .json("artifacts")
            .error_format("json")
            .incremental(cwd())
            .run();
        let stderr = output.stderr_utf8();
        for file in &["rcgu.o", "rcgu.ll", "rcgu.bc", "rcgu.s"] {
            assert!(stderr.contains(file), "No {:?} in {:?}", file, stderr);
        }
    }
}
