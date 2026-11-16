//! Check that abort library functions raise the expected signals.

//@ ignore-cross-compile

use run_make_support::{bin_name, run_fail_with_args, rustc};

#[derive(Debug)]
enum Variant {
    Abort,
    Crash,
}

fn main() {
    rustc().input("main.rs").run();

    check_one(Variant::Abort);
    check_one(Variant::Crash);
}

fn check_one(variant: Variant) {
    println!("checking variant {variant:?}");
    let arg = match variant {
        Variant::Abort => "abort",
        Variant::Crash => "crash",
    };
    let bin = &bin_name("main");
    let out = run_fail_with_args(bin, &[arg]);

    let status = out.status();
    println!("output status {status}");
    assert!(!status.success());

    cfg_select! {
        unix => {
            use std::assert_matches;
            use std::os::unix::process::ExitStatusExt;

            use run_make_support::libc;

            let sig = status.signal().expect("no signal recorded");
            match variant {
                Variant::Abort => assert_eq!(sig, libc::SIGABRT),
                Variant::Crash => assert_matches!(sig, libc::SIGILL | libc::SIGTRAP),
            };
        }
        windows => {
            use run_make_support::windows;
            use windows::Win32::Foundation::{
                STATUS_ILLEGAL_INSTRUCTION, STATUS_STACK_BUFFER_OVERRUN,
            };

            let code = status.code().unwrap();
            let expected = match variant {
                Variant::Abort => STATUS_STACK_BUFFER_OVERRUN.0,
                Variant::Crash => STATUS_ILLEGAL_INSTRUCTION.0,
            };
            assert_eq!(code, expected);
        }
        _ => {
            unimplemented!("target may need a new branch")
        }
    }
}
