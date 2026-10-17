//@ needs-target-std
use run_make_support::{path, rustc};

fn main() {
    rustc().input("foo.rs").emit("asm,llvm-ir").output("out").run();

    assert!(path("out.ll").is_file());
    assert!(path("out.s").is_file());

    rustc().input("foo.rs").emit("asm,llvm-ir").output("out2.ext").run();

    assert!(path("out2.ll").is_file());
    assert!(path("out2.s").is_file());
}
