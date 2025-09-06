//! This test is supposed to check that --emit=mir emits both optimized_mir and mir_for_ctfe for a
//! const fn.

use run_make_support::{diff, rustc};

fn main() {
    rustc().input("lib.rs").emit("mir").output("dump-actual.mir").run();
    diff().expected_file("dump.mir").actual_file("dump-actual.mir").run();
}
