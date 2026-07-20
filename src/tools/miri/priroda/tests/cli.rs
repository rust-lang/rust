use std::env;
use std::path::PathBuf;
use std::process::Command;

use regex::bytes::Regex;
use ui_test::spanned::Spanned;
use ui_test::status_emitter::StatusEmitter;
use ui_test::{CommandBuilder, Config, default_file_filter, run_tests_generic};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let miri_dir = manifest_dir.parent().unwrap();

    let rustc_sysroot = Command::new("rustc").arg("--print").arg("sysroot").output()?;
    let rustc_sysroot = String::from_utf8(rustc_sysroot.stdout)?.trim().to_owned();

    let mut program = CommandBuilder::rustc();
    program.program = PathBuf::from(env!("CARGO_BIN_EXE_priroda"));

    // Remove logging env vars that might leak into stderr
    program.envs.push(("RUSTC_LOG".into(), None));
    program.envs.push(("RUST_LOG".into(), None));

    let mut config = Config {
        program,
        out_dir: PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("priroda_ui"),
        ..Config::rustc("tests/ui")
    };

    // Replace the dynamic paths in the actual stdout with the stable placeholders
    let manifest_dir_regex =
        Regex::new(&regex::escape(&manifest_dir.display().to_string())).unwrap();
    let miri_dir_regex = Regex::new(&regex::escape(&miri_dir.display().to_string())).unwrap();
    let rustc_sysroot_regex = Regex::new(&regex::escape(&rustc_sysroot)).unwrap();
    let pointer_regex = Regex::new(r"0x[0-9a-f]+\[alloc[0-9]+\]<[0-9]+>").unwrap();
    config.comment_defaults.base().normalize_stdout.extend([
        (manifest_dir_regex.into(), b"{MANIFEST_DIR}".to_vec()),
        (miri_dir_regex.into(), b"{MIRI_DIR}".to_vec()),
        (rustc_sysroot_regex.into(), b"{RUSTC_SYSROOT}".to_vec()),
        (pointer_regex.into(), b"{ALLOC_PTR}".to_vec()),
    ]);

    // Priroda CLI tests do not currently require annotation comments in the test files
    config.comment_defaults.base().exit_status = Spanned::dummy(0).into();
    config.comment_defaults.base().require_annotations = Spanned::dummy(false).into();

    let mut args = ui_test::Args::test()?;
    args.bless |= env::var_os("RUSTC_BLESS").is_some_and(|v| v != "0");
    config.with_args(&args);

    run_tests_generic(
        vec![config],
        default_file_filter,
        |_, _| {},
        Box::<dyn StatusEmitter>::from(args.format),
    )?;

    Ok(())
}
