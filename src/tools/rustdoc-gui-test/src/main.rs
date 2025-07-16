use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use build_helper::npm;
use build_helper::util::try_run;
use compiletest::directives::TestProps;
use config::Config;

mod config;

fn find_librs<P: AsRef<Path>>(path: P) -> Option<PathBuf> {
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry.ok()?;
        if entry.file_type().is_file() && entry.file_name() == "lib.rs" {
            return Some(entry.path().to_path_buf());
        }
    }
    None
}

fn main() -> Result<(), ()> {
    let config = Arc::new(Config::from_args(env::args().collect()));

    let src_path = config.rust_src.join("tests/rustdoc-gui/src");
    for entry in src_path.read_dir().expect("read_dir call failed") {
        if let Ok(entry) = entry {
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let mut cargo = Command::new(&config.initial_cargo);
            cargo
                .arg("doc")
                .arg("--target-dir")
                .arg(&config.out_dir)
                .env("RUSTC_BOOTSTRAP", "1")
                .env("RUSTDOC", &config.rustdoc)
                .env("RUSTC", &config.rustc)
                .current_dir(path);

            if let Some(librs) = find_librs(entry.path()) {
                let compiletest_c = compiletest::common::Config::incomplete_for_rustdoc_gui_test();

                let test_props = TestProps::from_file(
                    &camino::Utf8PathBuf::try_from(librs).unwrap(),
                    None,
                    &compiletest_c,
                );

                if !test_props.compile_flags.is_empty() {
                    cargo.env("RUSTDOCFLAGS", test_props.compile_flags.join(" "));
                }

                cargo.args(&test_props.run_flags);
            }

            if try_run(&mut cargo, config.verbose).is_err() {
                eprintln!("failed to document `{}`", entry.path().display());
                panic!("Cannot run rustdoc-gui tests");
            }
        }
    }

    // FIXME(binarycat): once we get package.json in version control, this should be updated to install via that instead
    let local_node_modules =
        npm::install_one(&config.out_dir, &config.npm, "browser-ui-test", "0.21.1")
            .expect("unable to install browser-ui-test");

    let mut command = Command::new(&config.nodejs);

    command
        .arg(config.rust_src.join("src/tools/rustdoc-gui/tester.js"))
        .arg("--jobs")
        .arg(&config.jobs)
        .arg("--doc-folder")
        .arg(config.out_dir.join("doc"))
        .arg("--tests-folder")
        .arg(config.rust_src.join("tests/rustdoc-gui"));

    if local_node_modules.exists() {
        // Link the local node_modules if exists.
        // This is useful when we run rustdoc-gui-test from outside of the source root.
        command.env("NODE_PATH", local_node_modules);
    }

    for file in &config.goml_files {
        command.arg("--file").arg(file);
    }

    command.args(&config.test_args);

    try_run(&mut command, config.verbose)
}
