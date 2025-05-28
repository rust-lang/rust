use std::ffi::OsStr;
use std::path::Path;

use crate::utils::run_command_with_output;

fn show_usage() {
    println!(
        r#"
`fuzz` command help:
    --help                 : Show this help"#
    );
}

pub fn run() -> Result<(), String> {
    // We skip binary name and the `fuzz` command.
    let mut args = std::env::args().skip(2);
    let mut start = 0;
    let mut count = 100;
    let mut threads =
        std::thread::available_parallelism().map(|threads| threads.get()).unwrap_or(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => {
                show_usage();
                return Ok(());
            }
            "--start" => {
                start =
                    str::parse(&args.next().ok_or_else(|| "Fuzz start not provided!".to_string())?)
                        .map_err(|err| (format!("Fuzz start not a number {err:?}!")))?;
            }
            "--count" => {
                count =
                    str::parse(&args.next().ok_or_else(|| "Fuzz count not provided!".to_string())?)
                        .map_err(|err| (format!("Fuzz count not a number {err:?}!")))?;
            }
            "-j" | "--jobs" => {
                threads = str::parse(
                    &args.next().ok_or_else(|| "Fuzz thread count not provided!".to_string())?,
                )
                .map_err(|err| (format!("Fuzz thread count not a number {err:?}!")))?;
            }
            _ => return Err(format!("Unknown option {}", arg)),
        }
    }

    // Ensure that we have a cloned version of rustlantis on hand.
    crate::utils::git_clone(
        "https://github.com/cbeuw/rustlantis.git",
        Some("clones/rustlantis".as_ref()),
        true,
    )
    .map_err(|err| (format!("Git clone failed with message: {err:?}!")))?;

    // Ensure that we are on the newest rustlantis commit.
    let cmd: &[&dyn AsRef<OsStr>] = &[&"git", &"pull", &"origin"];
    run_command_with_output(cmd, Some(&Path::new("clones/rustlantis")))?;

    // Build the release version of rustlantis
    let cmd: &[&dyn AsRef<OsStr>] = &[&"cargo", &"build", &"--release"];
    run_command_with_output(cmd, Some(&Path::new("clones/rustlantis")))?;
    // Fuzz a given range
    fuzz_range(start, start + count, threads);
    Ok(())
}

/// Fuzzes a range `start..end` with `threads`.
fn fuzz_range(start: u64, end: u64, threads: usize) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};
    // Total amount of files to fuzz
    let total = end - start;
    // Currently fuzzed element
    let start = Arc::new(AtomicU64::new(start));
    // Count time during fuzzing
    let start_time = Instant::now();
    // Spawn `threads`..
    for _ in 0..threads {
        let start = start.clone();
        // .. which each will ..
        std::thread::spawn(move || {
            // ... grab the next fuzz seed ...
            while start.load(Ordering::Relaxed) < end {
                let next = start.fetch_add(1, Ordering::Relaxed);
                // .. test that seed .
                match test(next) {
                    Err(err) => {
                        // If the test failed at compile-time...
                        println!("test({}) failed because {err:?}", next);
                        // ... copy that file to the directory `target/fuzz/compiletime_error`...
                        let mut out_path: std::path::PathBuf =
                            "target/fuzz/compiletime_error".into();
                        std::fs::create_dir_all(&out_path).unwrap();
                        // .. into a file named `fuzz{seed}.rs`.
                        out_path.push(&format!("fuzz{next}.rs"));
                        std::fs::copy(err, out_path).unwrap();
                    }
                    Ok(Err(err)) => {
                        // If the test failed at run-time...
                        println!("The LLVM and GCC results don't match for {err:?}");
                        // ... copy that file to the directory `target/fuzz/runtime_error`...
                        let mut out_path: std::path::PathBuf = "target/fuzz/runtime_error".into();
                        std::fs::create_dir_all(&out_path).unwrap();
                        // .. into a file named `fuzz{seed}.rs`.
                        out_path.push(&format!("fuzz{next}.rs"));
                        std::fs::copy(err, out_path).unwrap();
                    }
                    // If the test passed, do nothing
                    Ok(Ok(())) => (),
                }
            }
        });
    }
    // The "manager" thread loop.
    while start.load(Ordering::Relaxed) < end {
        // Every 500 ms...
        let five_hundred_millis = Duration::from_millis(500);
        std::thread::sleep(five_hundred_millis);
        // ... calculate the remaining fuzz iters ...
        let remaining = end - start.load(Ordering::Relaxed);
        // ... fix the count(the start counter counts the cases that
        // begun fuzzing, and not only the ones that are done)...
        let fuzzed = (total - remaining) - threads as u64;
        // ... and the fuzz speed ...
        let iter_per_sec = fuzzed as f64 / start_time.elapsed().as_secs_f64();
        // .. and use them to display fuzzing stats.
        println!(
            "fuzzed {fuzzed} cases({}%), at rate {iter_per_sec} iter/s, remaining ~{}s",
            (100 * fuzzed) as f64 / total as f64,
            (remaining as f64) / iter_per_sec
        )
    }
}

/// Builds & runs a file with LLVM.
fn debug_llvm(path: &std::path::Path) -> Result<Vec<u8>, String> {
    // Build a file named `llvm_elf`...
    let exe_path = path.with_extension("llvm_elf");
    // ... using the LLVM backend ...
    let output = std::process::Command::new("rustc")
        .arg(path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .map_err(|err| format!("{err:?}"))?;
    // ... check that the compilation succeeded ...
    if !output.status.success() {
        return Err(format!("LLVM compilation failed:{output:?}"));
    }
    // ... run the resulting executable ...
    let output =
        std::process::Command::new(&exe_path).output().map_err(|err| format!("{err:?}"))?;
    // ... check it run normally ...
    if !output.status.success() {
        return Err(format!(
            "The program at {path:?}, compiled with LLVM, exited unsuccessfully:{output:?}"
        ));
    }
    // ... cleanup that executable ...
    std::fs::remove_file(exe_path).map_err(|err| format!("{err:?}"))?;
    // ... and return the output(stdout + stderr - this allows UB checks to fire).
    let mut res = output.stdout;
    res.extend(output.stderr);
    Ok(res)
}

/// Builds & runs a file with GCC.
fn release_gcc(path: &std::path::Path) -> Result<Vec<u8>, String> {
    // Build a file named `gcc_elf`...
    let exe_path = path.with_extension("gcc_elf");
    // ... using the GCC backend ...
    let output = std::process::Command::new("./y.sh")
        .arg("rustc")
        .arg(path)
        .arg("-O")
        .arg("-o")
        .arg(&exe_path)
        .output()
        .map_err(|err| format!("{err:?}"))?;
    // ... check that the compilation succeeded ...
    if !output.status.success() {
        return Err(format!("GCC compilation failed:{output:?}"));
    }
    // ... run the resulting executable ..
    let output =
        std::process::Command::new(&exe_path).output().map_err(|err| format!("{err:?}"))?;
    // ... check it run normally ...
    if !output.status.success() {
        return Err(format!(
            "The program at {path:?}, compiled with GCC, exited unsuccessfully:{output:?}"
        ));
    }
    // ... cleanup that executable ...
    std::fs::remove_file(exe_path).map_err(|err| format!("{err:?}"))?;
    // ... and return the output(stdout + stderr - this allows UB checks to fire).
    let mut res = output.stdout;
    res.extend(output.stderr);
    Ok(res)
}

/// Generates a new rustlantis file, & compares the result of running it with GCC and LLVM.
fn test(seed: u64) -> Result<Result<(), std::path::PathBuf>, String> {
    // Generate a Rust source...
    let source_file = generate(seed)?;
    // ... test it with debug LLVM ...
    let llvm_res = debug_llvm(&source_file)?;
    // ... test it with release GCC ...
    let gcc_res = release_gcc(&source_file)?;
    // ... compare the results ...
    if llvm_res != gcc_res {
        // .. if they don't match, report an error.
        Ok(Err(source_file))
    } else {
        std::fs::remove_file(source_file).map_err(|err| format!("{err:?}"))?;
        Ok(Ok(()))
    }
}

/// Generates a new rustlantis file for us to run tests on.
fn generate(seed: u64) -> Result<std::path::PathBuf, String> {
    use std::io::Write;
    let mut out_path = std::env::temp_dir();
    out_path.push(&format!("fuzz{seed}.rs"));
    // We need to get the command output here.
    let out = std::process::Command::new("cargo")
        .args(["run", "--release", "--bin", "generate"])
        .arg(&format!("{seed}"))
        .current_dir("clones/rustlantis")
        .output()
        .map_err(|err| format!("{err:?}"))?;
    // Stuff the rustlantis output in a source file.
    std::fs::File::create(&out_path)
        .map_err(|err| format!("{err:?}"))?
        .write_all(&out.stdout)
        .map_err(|err| format!("{err:?}"))?;
    Ok(out_path)
}
