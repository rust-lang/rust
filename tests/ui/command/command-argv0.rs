// run-pass

// ignore-windows - this is a unix-specific test
// ignore-emscripten no processes
// ignore-sgx no processes
use std::env;
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let args: Vec<_> = env::args().collect();

    if args.len() > 1 {
        assert_eq!(args[1], "doing-test");
        assert_eq!(args[0], "i have a silly name");

        println!("passed");
        return;
    }

    let output =
        Command::new(&args[0]).arg("doing-test").arg0("i have a silly name").output().unwrap();
    assert!(
        output.stderr.is_empty(),
        "Non-empty stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"passed\n");
}
