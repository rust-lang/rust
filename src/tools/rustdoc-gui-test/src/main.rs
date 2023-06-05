use build_helper::util::try_run;
use compiletest::header::TestProps;
use config::Config;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::{env, fs};

mod config;

fn get_browser_ui_test_version_inner(npm: &Path, global: bool) -> Option<String> {
    let mut command = Command::new(&npm);
    command.arg("list").arg("--parseable").arg("--long").arg("--depth=0");
    if global {
        command.arg("--global");
    }
    let lines = command
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or(String::new());
    lines
        .lines()
        .find_map(|l| l.split(':').nth(1)?.strip_prefix("browser-ui-test@"))
        .map(|v| v.to_owned())
}

fn get_browser_ui_test_version(npm: &Path) -> Option<String> {
    get_browser_ui_test_version_inner(npm, false)
        .or_else(|| get_browser_ui_test_version_inner(npm, true))
}

fn compare_browser_ui_test_version(installed_version: &str, src: &Path) {
    match fs::read_to_string(
        src.join("src/ci/docker/host-x86_64/x86_64-gnu-tools/browser-ui-test.version"),
    ) {
        Ok(v) => {
            if v.trim() != installed_version {
                eprintln!(
                    "⚠️ Installed version of browser-ui-test (`{}`) is different than the \
                     one used in the CI (`{}`)",
                    installed_version, v
                );
                eprintln!(
                    "You can install this version using `npm update browser-ui-test` or by using \
                     `npm install browser-ui-test@{}`",
                    v,
                );
            }
        }
        Err(e) => eprintln!("Couldn't find the CI browser-ui-test version: {:?}", e),
    }
}

fn find_librs<P: AsRef<Path>>(path: P) -> Option<PathBuf> {
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry.ok()?;
        if entry.file_type().is_file() && entry.file_name() == "lib.rs" {
            return Some(entry.path().to_path_buf());
        }
    }
    None
}

fn main() {
    let config = Arc::new(Config::from_args(env::args().collect()));

    // The goal here is to check if the necessary packages are installed, and if not, we
    // panic.
    match get_browser_ui_test_version(&config.npm) {
        Some(version) => {
            // We also check the version currently used in CI and emit a warning if it's not the
            // same one.
            compare_browser_ui_test_version(&version, &config.rust_src);
        }
        None => {
            eprintln!(
                r#"
error: rustdoc-gui test suite cannot be run because npm `browser-ui-test` dependency is missing.

If you want to install the `browser-ui-test` dependency, run `npm install browser-ui-test`
"#,
            );

            panic!("Cannot run rustdoc-gui tests");
        }
    }

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
                let compiletest_c = compiletest::common::Config {
                    edition: None,
                    mode: compiletest::common::Mode::Rustdoc,
                    ..Default::default()
                };

                let test_props = TestProps::from_file(&librs, None, &compiletest_c);

                if !test_props.compile_flags.is_empty() {
                    cargo.env("RUSTDOCFLAGS", test_props.compile_flags.join(" "));
                }

                if let Some(flags) = &test_props.run_flags {
                    cargo.arg(flags);
                }
            }

            try_run(&mut cargo, config.verbose);
        }
    }

    let mut command = Command::new(&config.nodejs);

    if let Ok(current_dir) = env::current_dir() {
        let local_node_modules = current_dir.join("node_modules");
        if local_node_modules.exists() {
            // Link the local node_modules if exists.
            // This is useful when we run rustdoc-gui-test from outside of the source root.
            env::set_var("NODE_PATH", local_node_modules);
        }
    }

    command
        .arg(config.rust_src.join("src/tools/rustdoc-gui/tester.js"))
        .arg("--jobs")
        .arg(&config.jobs)
        .arg("--doc-folder")
        .arg(config.out_dir.join("doc"))
        .arg("--tests-folder")
        .arg(config.rust_src.join("tests/rustdoc-gui"));

    for file in &config.goml_files {
        command.arg("--file").arg(file);
    }

    command.args(&config.test_args);

    try_run(&mut command, config.verbose);
}
