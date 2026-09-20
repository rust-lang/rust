use rustc_session::config::rustc_optgroups;
use rustc_session::getopts;

use crate::opts;

fn rustdoc_getopts() -> getopts::Options {
    let mut options = getopts::Options::new();
    for option in opts() {
        option.apply(&mut options);
    }
    options
}

#[test]
fn every_rustc_option_is_accepted_by_rustdoc() {
    let options = rustdoc_getopts();
    let unrecognized: Vec<String> = rustc_optgroups()
        .iter()
        .map(|opt| {
            let long = opt.long_name();
            if long.is_empty() { format!("-{}", opt.name) } else { format!("--{long}") }
        })
        .filter(|arg| {
            matches!(
                options.parse([arg.as_str(), "dummy"]),
                Err(getopts::Fail::UnrecognizedOption(_))
            )
        })
        .collect();
    assert!(unrecognized.is_empty(), "rustc options rejected by rustdoc: {unrecognized:?}");
}

#[test]
fn every_override_names_a_rustc_option() {
    let rustc_names: Vec<&str> = rustc_optgroups().iter().map(|opt| opt.name).collect();
    let stale: Vec<&str> =
        crate::RUSTDOC_OVERRIDES.iter().copied().filter(|n| !rustc_names.contains(n)).collect();
    assert!(stale.is_empty(), "overrides that no longer match a rustc option: {stale:?}");
}

#[test]
fn options_are_declared_once() {
    let mut names: Vec<&str> = opts().iter().map(|opt| opt.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate option declarations: {names:?}");
}

#[test]
fn forwarded_rustc_args_preserve_command_line_order() {
    let matches = rustdoc_getopts()
        .parse([
            "-Copt-level=1",
            "--cfg",
            "a",
            "-O",
            "-Zfoo",
            "-lbar",
            "-Lx",
            "--library-path=y",
            "--extern",
            "e=path",
            "-g",
            "--codegen",
            "debuginfo=1",
            "--check-cfg",
            "cfg(a)",
            "--out-dir",
            "ignored",
            "input.rs",
        ])
        .unwrap();
    assert_eq!(
        super::forwarded_rustc_args(&matches),
        [
            "-Copt-level=1",
            "--cfg=a",
            "-O",
            "-Zfoo",
            "-lbar",
            "-Lx",
            "-Ly",
            "--extern=e=path",
            "-g",
            "-Cdebuginfo=1",
            "--check-cfg=cfg(a)",
        ]
    );
}
