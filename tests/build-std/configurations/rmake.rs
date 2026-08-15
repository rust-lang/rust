// This test ensures we are able to compile -Zbuild-std=core under a variety of profiles.
// Currently, it tests that we can compile to all Tier 1 targets, and it does this by checking what
// the tier metadata in target-spec JSON. This means that all in-tree targets must have a tier set.

#![deny(warnings)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use run_make_support::serde_json::{self, Value};
use run_make_support::tempfile::TempDir;
use run_make_support::{cargo, rfs, rustc};

#[derive(Clone)]
struct Task {
    target: String,
    opt_level: u8,
    debug: u8,
    panic: &'static str,
}

fn manifest(task: &Task) -> String {
    let Task { opt_level, debug, panic, target: _ } = task;
    format!(
        r#"[package]
name = "scratch"
version = "0.1.0"
edition = "2024"

[lib]
path = "lib.rs"

[profile.release]
opt-level = {opt_level}
debug = {debug}
panic = "{panic}"
"#
    )
}

fn main() {
    let mut targets = Vec::new();
    let all_targets =
        rustc().args(&["--print=all-target-specs-json", "-Zunstable-options"]).run().stdout_utf8();
    let all_targets: HashMap<String, Value> = serde_json::from_str(&all_targets).unwrap();
    for (target, spec) in all_targets {
        let metadata = spec.as_object().unwrap()["metadata"].as_object().unwrap();
        let tier = metadata["tier"]
            .as_u64()
            .expect(&format!("Target {} is missing tier metadata", target));
        if tier == 1 {
            targets.push(target);
        }
    }

    let mut tasks = Vec::new();

    // Testing every combination of compiler flags is infeasible. So we are making some attempt to
    // choose combinations that will tend to run into problems.
    //
    // The particular combination of settings below is tuned to look for problems generating the
    // code for compiler-builtins.
    // We only exercise opt-level 0 and 3 to exercise mir-opt-level 1 and 2.
    // We only exercise debug 0 and 2 because level 2 turns off some MIR optimizations.
    // We only test abort and immediate-abort because abort vs unwind doesn't change MIR much at
    // all. but immediate-abort does.
    //
    // Currently this only tests that we can compile the tier 1 targets. But since we are using
    // -Zbuild-std=core, we could have any list of targets.

    for opt_level in [0, 3] {
        for debug in [0, 2] {
            for panic in ["abort", "immediate-abort"] {
                for target in &targets {
                    tasks.push(Task { target: target.clone(), opt_level, debug, panic });
                }
            }
        }
    }

    let tasks = Arc::new(Mutex::new(tasks));
    let mut threads = Vec::new();

    // Try to obey the -j argument passed to bootstrap, otherwise fall back to using all the system
    // resouces. This test can be rather memory-hungry (~1 GB/thread); if it causes trouble in
    // practice do not hesitate to limit its parallelism.
    for _ in 0..run_make_support::env::jobs() {
        let tasks = Arc::clone(&tasks);
        let handle = thread::spawn(move || {
            loop {
                let maybe_task = tasks.lock().unwrap().pop();
                if let Some(task) = maybe_task {
                    test(task);
                } else {
                    break;
                }
            }
        });
        threads.push(handle);
    }

    for t in threads {
        t.join().unwrap();
    }
}

fn test(task: Task) {
    let dir = TempDir::new().unwrap();

    let manifest = manifest(&task);
    rfs::write(dir.path().join("Cargo.toml"), &manifest);
    rfs::write(dir.path().join("lib.rs"), "#![no_std]");

    let mut args = vec!["build", "--release", "-Zbuild-std=core", "--target", &task.target, "-j1"];
    if task.panic == "immediate-abort" {
        args.push("-Zpanic-immediate-abort");
    }
    cargo()
        .current_dir(dir.path())
        .args(&args)
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .context(&format!(
            "build-std for target `{}` failed with the following Cargo.toml:\n\n{manifest}",
            task.target
        ))
        .run();
}
