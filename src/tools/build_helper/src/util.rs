use std::process::Command;

/// Invokes `build_helper::util::detail_exit` with `cfg!(test)`
#[macro_export]
macro_rules! detail_exit_macro {
    ($code:expr) => {
        build_helper::util::detail_exit($code, cfg!(test));
    };
}

/// If code is not 0 (successful exit status), exit status is 101 (rust's default error code.)
/// If `is_test` true and code is an error code, it will cause a panic.
pub fn detail_exit(code: i32, is_test: bool) -> ! {
    // if in test and code is an error code, panic with status code provided
    if is_test {
        panic!("status code: {}", code);
    } else {
        // otherwise,exit with provided status code
        std::process::exit(code);
    }
}

pub fn fail(s: &str) -> ! {
    eprintln!("\n\n{}\n\n", s);
    detail_exit(1, cfg!(test));
}

pub fn try_run(cmd: &mut Command, print_cmd_on_fail: bool) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !status.success() && print_cmd_on_fail {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n",
            cmd, status
        );
    }
    status.success()
}
