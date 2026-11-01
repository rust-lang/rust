// The `needs-unwind -Cpanic=abort` gives a different MIR output.

//@ needs-unwind

use run_make_support::{diff, rustc};

fn main() {
    rustc().input("main.rs").emit("mir").output("dump-actual.mir").run();
    diff().expected_file("dump.mir").actual_file("dump-actual.mir").run();
}
