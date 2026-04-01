//@ needs-target-std
use std::path::Path;

use run_make_support::{rfs, rustc};

fn emit_and_check(out_dir: &Path, out_file: &str, format: &str) {
    let out_file = out_dir.join(out_file);
    rustc().input("foo.rs").emit(format!("{format}={}", out_file.display())).run();
    assert!(out_file.is_file());
}

fn main() {
    let out_dir = Path::new("emit");

    rfs::create_dir(&out_dir);

    emit_and_check(&out_dir, "libfoo.s", "asm");
    emit_and_check(&out_dir, "libfoo.bc", "llvm-bc");
    emit_and_check(&out_dir, "libfoo.ll", "llvm-ir");
    emit_and_check(&out_dir, "libfoo.o", "obj");
    emit_and_check(&out_dir, "libfoo.rmeta", "metadata");
    emit_and_check(&out_dir, "libfoo.rlib", "link");
    emit_and_check(&out_dir, "libfoo.d", "dep-info");
    emit_and_check(&out_dir, "libfoo.mir", "mir");
}
