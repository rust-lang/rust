// A bug from 2015 would cause errors when emitting multiple types of files
// in the same rustc call. A fix was created in #30452. This test checks that rustc still compiles
// a source file successfully when emission of multiple output artifacts are requested.
// See https://github.com/rust-lang/rust/pull/30452

//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    let opt_levels = ["0", "1", "2", "3", "s", "z"];
    for level in opt_levels {
        rustc().opt_level(level).emit("llvm-bc,llvm-ir,asm,obj,link").input("test-24876.rs").run();
    }
    for level in opt_levels {
        rustc().opt_level(level).emit("llvm-bc,llvm-ir,asm,obj,link").input("test-26235.rs").run();
        run("test-26235");
    }
}
