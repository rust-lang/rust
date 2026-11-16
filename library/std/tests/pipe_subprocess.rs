#![feature(test)]

use std::io::{Read, pipe};
use std::process::ExitCode;
use std::{env, process};

// We invoke the test harness manually so that the same binary can also serve as the child
// process that sits on the other end of the pipe.
extern crate test;

static TESTS: &[&test::TestDescAndFn] = &[&test::TestDescAndFn {
    testfn: test::TestFn::StaticTestFn(pipe_subprocess),
    desc: test::TestDesc {
        name: test::StaticTestName("pipe_subprocess"),
        ignore: false,
        ignore_message: None,
        source_file: file!(),
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        compile_fail: false,
        no_run: false,
        should_panic: test::ShouldPanic::No,
        test_type: test::TestType::IntegrationTest,
    },
}];

fn parent() {
    let me = env::current_exe().unwrap();
    // If a custom `runner` is set up for the current target, we'll be
    // executing `./runner ./test`, not just `./test`. For such a case,
    // use the same arguments for child to avoid executing `runner`
    // without an actual executable.
    let args = env::args();

    let (rx, tx) = pipe().unwrap();
    assert!(
        process::Command::new(me)
            .args(args)
            .env("I_AM_THE_CHILD", "1")
            .stdout(tx)
            .status()
            .unwrap()
            .success()
    );

    let mut s = String::new();
    (&rx).read_to_string(&mut s).unwrap();
    drop(rx);
    assert_eq!(s, "Heloo,\n");

    println!("Test pipe_subprocess.rs success");
}

fn child() {
    println!("Heloo,");
}

fn pipe_subprocess() -> Result<(), String> {
    // No `Command` on Miri, emscripten or L4Re
    if cfg!(all(
        not(skip_unsupported_miri_tests),
        any(unix, windows),
        not(any(target_os = "emscripten", target_os = "l4re"))
    )) {
        parent();
    }
    Ok(())
}

fn main() -> ExitCode {
    if env::var("I_AM_THE_CHILD").is_ok() {
        child();
        ExitCode::SUCCESS
    } else {
        // Go via libtest runner so test filtering and nextest work properly.
        test::test_main_env_args(TESTS)
    }
}
