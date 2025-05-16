// Run clippy on a fixed set of crates and collect the warnings.
// This helps observing the impact clippy changes have on a set of real-world code (and not just our
// testsuite).
//
// When a new lint is introduced, we can search the results for new warnings and check for false
// positives.

#![feature(iter_collect_into)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]
#![allow(
    clippy::collapsible_else_if,
    clippy::needless_borrows_for_generic_args,
    clippy::module_name_repetitions,
    clippy::literal_string_with_formatting_args
)]

mod config;
mod driver;
mod input;
mod json;
mod output;
mod popular_crates;
mod recursive;

use crate::config::{Commands, LintcheckConfig, OutputFormat};
use crate::recursive::LintcheckServer;

use std::env::consts::EXE_SUFFIX;
use std::io::{self};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, fs};

use cargo_metadata::Message;
use input::read_crates;
use output::{ClippyCheckOutput, ClippyWarning, RustcIce};
use rayon::prelude::*;

const LINTCHECK_DOWNLOADS: &str = "target/lintcheck/downloads";
const LINTCHECK_SOURCES: &str = "target/lintcheck/sources";

/// Represents the actual source code of a crate that we ran "cargo clippy" on
#[derive(Debug)]
struct Crate {
    version: String,
    name: String,
    // path to the extracted sources that clippy can check
    path: PathBuf,
    options: Option<Vec<String>>,
    base_url: String,
}

impl Crate {
    /// Run `cargo clippy` on the `Crate` and collect and return all the lint warnings that clippy
    /// issued
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn run_clippy_lints(
        &self,
        clippy_driver_path: &Path,
        target_dir_index: &AtomicUsize,
        total_crates_to_lint: usize,
        config: &LintcheckConfig,
        lint_levels_args: &[String],
        server: Option<&LintcheckServer>,
    ) -> Vec<ClippyCheckOutput> {
        // advance the atomic index by one
        let index = target_dir_index.fetch_add(1, Ordering::SeqCst);
        // "loop" the index within 0..thread_limit
        let thread_index = index % config.max_jobs;
        let perc = (index * 100) / total_crates_to_lint;

        if config.max_jobs == 1 {
            println!(
                "{index}/{total_crates_to_lint} {perc}% Linting {} {}",
                &self.name, &self.version
            );
        } else {
            println!(
                "{index}/{total_crates_to_lint} {perc}% Linting {} {} in target dir {thread_index:?}",
                &self.name, &self.version
            );
        }

        let cargo_home = env!("CARGO_HOME");

        // `src/lib.rs` -> `target/lintcheck/sources/crate-1.2.3/src/lib.rs`
        let remap_relative = format!("={}", self.path.display());
        // Fallback for other sources, `~/.cargo/...` -> `$CARGO_HOME/...`
        let remap_cargo_home = format!("{cargo_home}=$CARGO_HOME");
        // `~/.cargo/registry/src/index.crates.io-6f17d22bba15001f/crate-2.3.4/src/lib.rs`
        //     -> `crate-2.3.4/src/lib.rs`
        let remap_crates_io = format!("{cargo_home}/registry/src/index.crates.io-6f17d22bba15001f/=");

        let mut clippy_args = vec![
            "--remap-path-prefix",
            &remap_relative,
            "--remap-path-prefix",
            &remap_cargo_home,
            "--remap-path-prefix",
            &remap_crates_io,
        ];

        if let Some(options) = &self.options {
            for opt in options {
                clippy_args.push(opt);
            }
        }

        clippy_args.extend(lint_levels_args.iter().map(String::as_str));

        let mut cmd;

        if config.perf {
            cmd = Command::new("perf");
            let perf_data_filename = get_perf_data_filename(&self.path);
            cmd.args(&[
                "record",
                "-e",
                "instructions", // Only count instructions
                "-g",           // Enable call-graph, useful for flamegraphs and produces richer reports
                "--quiet",      // Do not tamper with lintcheck's normal output
                "--compression-level=22",
                "--freq=3000", // Slow down program to capture all events
                "-o",
                &perf_data_filename,
                "--",
                "cargo",
            ]);
        } else {
            cmd = Command::new("cargo");
        }

        cmd.arg(if config.fix { "fix" } else { "check" })
            .arg("--quiet")
            .current_dir(&self.path)
            .env("CLIPPY_ARGS", clippy_args.join("__CLIPPY_HACKERY__"))
            .env("CLIPPY_DISABLE_DOCS_LINKS", "1");

        if let Some(server) = server {
            // `cargo clippy` is a wrapper around `cargo check` that mainly sets `RUSTC_WORKSPACE_WRAPPER` to
            // `clippy-driver`. We do the same thing here with a couple changes:
            //
            // `RUSTC_WRAPPER` is used instead of `RUSTC_WORKSPACE_WRAPPER` so that we can lint all crate
            // dependencies rather than only workspace members
            //
            // The wrapper is set to `lintcheck` itself so we can force enable linting and ignore certain crates
            // (see `crate::driver`)
            let status = cmd
                .env("CARGO_TARGET_DIR", shared_target_dir("recursive"))
                .env("RUSTC_WRAPPER", env::current_exe().unwrap())
                // Pass the absolute path so `crate::driver` can find `clippy-driver`, as it's executed in various
                // different working directories
                .env("CLIPPY_DRIVER", clippy_driver_path)
                .env("LINTCHECK_SERVER", server.local_addr.to_string())
                .status()
                .expect("failed to run cargo");

            assert_eq!(status.code(), Some(0));

            return Vec::new();
        }

        if !config.fix && !config.perf {
            cmd.arg("--message-format=json");
        }

        let shared_target_dir = shared_target_dir(&format!("_{thread_index:?}"));
        let all_output = cmd
            // use the looping index to create individual target dirs
            .env("CARGO_TARGET_DIR", shared_target_dir.as_os_str())
            // Roughly equivalent to `cargo clippy`/`cargo clippy --fix`
            .env("RUSTC_WORKSPACE_WRAPPER", clippy_driver_path)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&all_output.stdout);
        let stderr = String::from_utf8_lossy(&all_output.stderr);
        let status = &all_output.status;

        if !status.success() {
            eprintln!(
                "\nWARNING: bad exit status after checking {} {} \n",
                self.name, self.version
            );
        }

        if config.fix {
            if let Some(stderr) = stderr
                .lines()
                .find(|line| line.contains("failed to automatically apply fixes suggested by rustc to crate"))
            {
                let subcrate = &stderr[63..];
                println!(
                    "ERROR: failed to apply some suggestion to {} / to (sub)crate {subcrate}",
                    self.name
                );
            }
            // fast path, we don't need the warnings anyway
            return Vec::new();
        }

        // We don't want to keep target directories if benchmarking
        if config.perf {
            let _ = fs::remove_dir_all(&shared_target_dir);
        }

        // get all clippy warnings and ICEs
        let mut entries: Vec<ClippyCheckOutput> = Message::parse_stream(stdout.as_bytes())
            .filter_map(|msg| match msg {
                Ok(Message::CompilerMessage(message)) => ClippyWarning::new(
                    normalize_diag(message.message, shared_target_dir.to_str().unwrap()),
                    &self.base_url,
                    &self.name,
                ),
                _ => None,
            })
            .map(ClippyCheckOutput::ClippyWarning)
            .collect();

        if let Some(ice) = RustcIce::from_stderr_and_status(&self.name, *status, &stderr) {
            entries.push(ClippyCheckOutput::RustcIce(ice));
        } else if !status.success() {
            println!("non-ICE bad exit status for {} {}: {}", self.name, self.version, stderr);
        }

        entries
    }
}

/// The target directory can sometimes be stored in the file name of spans.
/// This is problematic since the directory in constructed from the thread
/// ID and also used in our CI to determine if two lint emissions are the
/// same or not. This function simply normalizes the `_<thread_id>` to `_*`.
fn normalize_diag(
    mut message: cargo_metadata::diagnostic::Diagnostic,
    thread_target_dir: &str,
) -> cargo_metadata::diagnostic::Diagnostic {
    let mut dir_found = false;
    message
        .spans
        .iter_mut()
        .filter(|span| span.file_name.starts_with(thread_target_dir))
        .for_each(|span| {
            dir_found = true;
            span.file_name
                .replace_range(0..thread_target_dir.len(), shared_target_dir("_*").to_str().unwrap());
        });

    if dir_found && let Some(rendered) = &mut message.rendered {
        *rendered = rendered.replace(thread_target_dir, shared_target_dir("_*").to_str().unwrap());
    }
    message
}

/// Builds clippy inside the repo to make sure we have a clippy executable we can use.
fn build_clippy(release_build: bool) -> String {
    let mut build_cmd = Command::new("cargo");
    build_cmd.args([
        "run",
        "--bin=clippy-driver",
        if release_build { "-r" } else { "" },
        "--",
        "--version",
    ]);

    if release_build {
        build_cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "true");
    }

    let output = build_cmd.stderr(Stdio::inherit()).output().unwrap();

    if !output.status.success() {
        eprintln!("Error: Failed to compile Clippy!");
        std::process::exit(1);
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn main() {
    // We're being executed as a `RUSTC_WRAPPER` as part of `--recursive`
    if let Ok(addr) = env::var("LINTCHECK_SERVER") {
        driver::drive(&addr);
    }

    // assert that we launch lintcheck from the repo root (via cargo lintcheck)
    if fs::metadata("lintcheck/Cargo.toml").is_err() {
        eprintln!("lintcheck needs to be run from clippy's repo root!\nUse `cargo lintcheck` alternatively.");
        std::process::exit(3);
    }

    let config = LintcheckConfig::new();

    match config.subcommand {
        Some(Commands::Diff { old, new, truncate }) => json::diff(&old, &new, truncate),
        Some(Commands::Popular { output, number }) => popular_crates::fetch(output, number).unwrap(),
        None => lintcheck(config),
    }
}

#[allow(clippy::too_many_lines)]
fn lintcheck(config: LintcheckConfig) {
    let clippy_ver = build_clippy(config.perf);
    let clippy_driver_path = fs::canonicalize(format!(
        "target/{}/clippy-driver{EXE_SUFFIX}",
        if config.perf { "release" } else { "debug" }
    ))
    .unwrap();

    // assert that clippy is found
    assert!(
        clippy_driver_path.is_file(),
        "target/{}/clippy-driver binary not found! {}",
        if config.perf { "release" } else { "debug" },
        clippy_driver_path.display()
    );

    // download and extract the crates, then run clippy on them and collect clippy's warnings
    // flatten into one big list of warnings

    let (crates, recursive_options) = read_crates(&config.sources_toml_path);

    let counter = AtomicUsize::new(1);
    let mut lint_level_args: Vec<String> = vec!["--cap-lints=allow".into()];
    if config.lint_filter.is_empty() {
        let groups = if config.all_lints {
            &[
                "clippy::all",
                "clippy::cargo",
                "clippy::nursery",
                "clippy::pedantic",
                "clippy::restriction",
            ][..]
        } else {
            &["clippy::all", "clippy::pedantic"]
        };
        groups
            .iter()
            .map(|group| format!("--force-warn={group}"))
            .collect_into(&mut lint_level_args);
    } else {
        config
            .lint_filter
            .iter()
            .map(|filter| {
                let mut filter = filter.clone();
                filter.insert_str(0, "--force-warn=");
                filter
            })
            .collect_into(&mut lint_level_args);
    }

    let crates: Vec<Crate> = crates
        .into_iter()
        .filter(|krate| {
            if let Some(only_one_crate) = &config.only {
                krate.name == *only_one_crate
            } else {
                true
            }
        })
        .map(|krate| krate.download_and_prepare())
        .collect();

    if crates.is_empty() {
        eprintln!(
            "ERROR: could not find crate '{}' in lintcheck/lintcheck_crates.toml",
            config.only.unwrap(),
        );
        std::process::exit(1);
    }

    // run parallel with rayon

    // This helps when we check many small crates with dep-trees that don't have a lot of branches in
    // order to achieve some kind of parallelism

    rayon::ThreadPoolBuilder::new()
        .num_threads(config.max_jobs)
        .build_global()
        .unwrap();

    let server = config.recursive.then(|| {
        let _: io::Result<()> = fs::remove_dir_all("target/lintcheck/shared_target_dir/recursive");

        LintcheckServer::spawn(recursive_options)
    });

    let mut clippy_entries: Vec<ClippyCheckOutput> = crates
        .par_iter()
        .flat_map(|krate| {
            krate.run_clippy_lints(
                &clippy_driver_path,
                &counter,
                crates.len(),
                &config,
                &lint_level_args,
                server.as_ref(),
            )
        })
        .collect();

    if let Some(server) = server {
        let server_clippy_entries = server.warnings().map(ClippyCheckOutput::ClippyWarning);

        clippy_entries.extend(server_clippy_entries);
    }

    // if we are in --fix mode, don't change the log files, terminate here
    if config.fix {
        return;
    }

    // split up warnings and ices
    let mut warnings: Vec<ClippyWarning> = vec![];
    let mut raw_ices: Vec<RustcIce> = vec![];
    for entry in clippy_entries {
        if let ClippyCheckOutput::ClippyWarning(x) = entry {
            warnings.push(x);
        } else if let ClippyCheckOutput::RustcIce(x) = entry {
            raw_ices.push(x);
        }
    }

    let text = match config.format {
        OutputFormat::Text | OutputFormat::Markdown => {
            output::summarize_and_print_changes(&warnings, &raw_ices, clippy_ver, &config)
        },
        OutputFormat::Json => {
            if !raw_ices.is_empty() {
                for ice in raw_ices {
                    println!("{ice}");
                }
                panic!("Some crates ICEd");
            }

            json::output(warnings)
        },
    };

    println!("Writing logs to {}", config.lintcheck_results_path.display());
    fs::create_dir_all(config.lintcheck_results_path.parent().unwrap()).unwrap();
    fs::write(&config.lintcheck_results_path, text).unwrap();
}

/// Traverse a directory looking for `perf.data.<number>` files, and adds one
/// to the most recent of those files, returning the new most recent `perf.data`
/// file name.
fn get_perf_data_filename(source_path: &Path) -> String {
    if source_path.join("perf.data").exists() {
        let mut max_number = 0;
        fs::read_dir(source_path)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|path| {
                path.file_name()
                    .as_os_str()
                    .to_string_lossy() // We don't care about data loss, as we're checking for equality
                    .starts_with("perf.data")
            })
            .for_each(|path| {
                let file_name = path.file_name();
                let file_name = file_name.as_os_str().to_str().unwrap().split('.').next_back().unwrap();
                if let Ok(parsed_file_name) = file_name.parse::<usize>()
                    && parsed_file_name >= max_number
                {
                    max_number = parsed_file_name + 1;
                }
            });
        return format!("perf.data.{max_number}");
    }
    String::from("perf.data")
}

/// Returns the path to the Clippy project directory
#[must_use]
fn clippy_project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

/// The qualifier can be used to separate different threads from another. By
/// default it should be set to `_<thread_id>`
#[must_use]
fn shared_target_dir(qualifier: &str) -> PathBuf {
    clippy_project_root()
        .join("target/lintcheck/shared_target_dir")
        .join(qualifier)
}

#[test]
fn lintcheck_test() {
    let args = [
        "run",
        "--target-dir",
        "lintcheck/target",
        "--manifest-path",
        "./lintcheck/Cargo.toml",
        "--",
        "--crates-toml",
        "lintcheck/test_sources.toml",
    ];
    let status = Command::new("cargo")
        .args(args)
        .current_dir("..") // repo root
        .status();
    //.output();

    assert!(status.unwrap().success());
}
