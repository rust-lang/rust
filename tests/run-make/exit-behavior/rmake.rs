//! Check exit codes and signals of terminating library functions.

//@ ignore-cross-compile

use run_make_support::{bin_name, run_fail_with_args, run_with_args, rustc};

#[derive(Debug)]
enum Variant {
    Abort,
    Crash,
    Panic,
    Exit0,
    Exit1,
}

fn main() {
    rustc().input("main.rs").edition("2024").run();

    check_one(Variant::Abort);
    check_one(Variant::Crash);
    check_one(Variant::Panic);
    check_one(Variant::Exit0);
    check_one(Variant::Exit1);
}

fn check_one(variant: Variant) {
    println!("checking variant {variant:?}");
    let arg = match variant {
        Variant::Abort => "abort",
        Variant::Crash => "crash",
        Variant::Panic => "panic",
        Variant::Exit0 => "exit-0",
        Variant::Exit1 => "exit-1",
    };
    let bin = &bin_name("main");

    let (out, expected_success) = if matches!(variant, Variant::Exit0) {
        (run_with_args(bin, &[arg]), true)
    } else {
        (run_fail_with_args(bin, &[arg]), false)
    };

    let status = out.status();
    let code = status.code();
    println!("output status for {variant:?}: {status:?}");
    assert_eq!(status.success(), expected_success);

    cfg_select! {
        unix => {
            use std::assert_matches;
            use std::os::unix::process::ExitStatusExt;

            use run_make_support::libc;

            let expected_code = match variant {
                Variant::Abort => None,
                Variant::Crash => None,
                Variant::Panic => Some(101),
                Variant::Exit0 => Some(0),
                Variant::Exit1 => Some(1),
            };
            assert_eq!(code, expected_code);

            let sig = status.signal();
            match variant {
                Variant::Abort => assert_eq!(sig, Some(libc::SIGABRT)),
                Variant::Crash => assert_matches!(sig, Some(libc::SIGILL | libc::SIGTRAP)),
                Variant::Panic => assert_eq!(sig, None),
                Variant::Exit0 => assert_eq!(sig, None),
                Variant::Exit1 => assert_eq!(sig, None),
            };
        }
        windows => {
            use run_make_support::windows;
            use windows::Win32::Foundation::{
                STATUS_ILLEGAL_INSTRUCTION, STATUS_STACK_BUFFER_OVERRUN,
            };

            let code = code.expect("Windows always has an exit code");
            let expected_code = match variant {
                Variant::Abort => STATUS_STACK_BUFFER_OVERRUN.0,
                Variant::Crash => STATUS_ILLEGAL_INSTRUCTION.0,
                Variant::Panic => 101,
                Variant::Exit0 => 0,
                Variant::Exit1 => 1,
            };
            assert_eq!(code, expected_code);
        }
        _ => {
            unimplemented!("target may need a new branch")
        }
    }
}
