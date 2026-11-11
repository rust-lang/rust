//@ only-arm64e-apple-darwin

use run_make_support::{cc, run, rustc};

fn main() {
    cc().arg("-c").arg("-O0").input("helper.c").output("helper.o").run();
    for opt in ["0", "3"] {
        let exe = format!("arm64e-o{opt}");
        rustc()
            .input("main.rs")
            .opt_level(opt)
            .arg("-Clink-arg=helper.o")
            .arg("-Clink-arg=-Wl,-U,_arm64e_missing_function")
            .output(&exe)
            .run();
        run(&exe).assert_stdout_equals("arm64e pointer authentication: PASS\n");
    }
}
