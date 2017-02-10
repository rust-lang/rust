extern crate compiletest_rs as compiletest;

use std::path::{PathBuf, Path};
use std::io::Write;

fn compile_fail(sysroot: &Path) {
    let flags = format!("--sysroot {} -Dwarnings", sysroot.to_str().expect("non utf8 path"));
    for_all_targets(&sysroot, |target| {
        let mut config = compiletest::default_config();
        config.host_rustcflags = Some(flags.clone());
        config.mode = "compile-fail".parse().expect("Invalid mode");
        config.run_lib_path = Path::new(sysroot).join("lib").join("rustlib").join(&target).join("lib");
        config.rustc_path = "target/debug/miri".into();
        config.src_base = PathBuf::from("tests/compile-fail".to_string());
        config.target = target.to_owned();
        config.target_rustcflags = Some(flags.clone());
        compiletest::run_tests(&config);
    });
}

fn run_pass() {
    let mut config = compiletest::default_config();
    config.mode = "run-pass".parse().expect("Invalid mode");
    config.src_base = PathBuf::from("tests/run-pass".to_string());
    config.target_rustcflags = Some("-Dwarnings".to_string());
    config.host_rustcflags = Some("-Dwarnings".to_string());
    compiletest::run_tests(&config);
}

fn miri_pass(path: &str, target: &str, host: &str) {
    let mut config = compiletest::default_config();
    config.mode = "mir-opt".parse().expect("Invalid mode");
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.host = host.to_owned();
    config.rustc_path = PathBuf::from("target/debug/miri");
    // don't actually execute the final binary, it might be for other targets and we only care
    // about running miri, not the binary.
    config.runtool = Some("echo \"\" || ".to_owned());
    if target == host {
        std::env::set_var("MIRI_HOST_TARGET", "yes");
    }
    compiletest::run_tests(&config);
    std::env::set_var("MIRI_HOST_TARGET", "");
}

fn is_target_dir<P: Into<PathBuf>>(path: P) -> bool {
    let mut path = path.into();
    path.push("lib");
    path.metadata().map(|m| m.is_dir()).unwrap_or(false)
}

fn for_all_targets<F: FnMut(String)>(sysroot: &Path, mut f: F) {
    let target_dir = sysroot.join("lib").join("rustlib");
    println!("target dir: {}", target_dir.to_str().unwrap());
    for entry in std::fs::read_dir(target_dir).expect("invalid sysroot") {
        let entry = entry.unwrap();
        if !is_target_dir(entry.path()) { continue; }
        let target = entry.file_name().into_string().unwrap();
        let stderr = std::io::stderr();
        writeln!(stderr.lock(), "running tests for target {}", target).unwrap();
        f(target);
    }
}

#[test]
fn compile_test() {
    let sysroot = std::process::Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("rustc not found")
        .stdout;
    let sysroot = std::str::from_utf8(&sysroot).expect("sysroot is not utf8").trim();
    let sysroot = &Path::new(&sysroot);
    let host = std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("rustc not found for -vV")
        .stdout;
    let host = std::str::from_utf8(&host).expect("sysroot is not utf8");
    let host = host.split("\nhost: ").skip(1).next().expect("no host: part in rustc -vV");
    let host = host.split("\n").next().expect("no \n after host");

    if let Ok(path) = std::env::var("MIRI_RUSTC_TEST") {
        let mut mir_not_found = Vec::new();
        let mut crate_not_found = Vec::new();
        let mut success = 0;
        let mut failed = Vec::new();
        let mut c_abi_fns = Vec::new();
        let mut abi = Vec::new();
        let mut unsupported = Vec::new();
        let mut unimplemented_intrinsic = Vec::new();
        let mut limits = Vec::new();
        for file in std::fs::read_dir(path).unwrap() {
            let file = file.unwrap();
            let path = file.path();
            if !file.metadata().unwrap().is_file() || !path.to_str().unwrap().ends_with(".rs") {
                continue;
            }
            let stderr = std::io::stderr();
            write!(stderr.lock(), "test [miri-pass] {} ... ", path.display()).unwrap();
            let mut cmd = std::process::Command::new("target/debug/miri");
            cmd.arg(path);
            let libs = Path::new(&sysroot).join("lib");
            let sysroot = libs.join("rustlib").join(&host).join("lib");
            let paths = std::env::join_paths(&[libs, sysroot]).unwrap();
            cmd.env(compiletest::procsrv::dylib_env_var(), paths);

            match cmd.output() {
                Ok(ref output) if output.status.success() => {
                    success += 1;
                    writeln!(stderr.lock(), "ok").unwrap()
                },
                Ok(output) => {
                    let output_err = std::str::from_utf8(&output.stderr).unwrap();
                    if let Some(text) = output_err.splitn(2, "no mir for `").nth(1) {
                        let end = text.find('`').unwrap();
                        mir_not_found.push(text[..end].to_string());
                        writeln!(stderr.lock(), "NO MIR FOR `{}`", &text[..end]).unwrap();
                    } else if let Some(text) = output_err.splitn(2, "can't find crate for `").nth(1) {
                        let end = text.find('`').unwrap();
                        crate_not_found.push(text[..end].to_string());
                        writeln!(stderr.lock(), "CAN'T FIND CRATE FOR `{}`", &text[..end]).unwrap();
                    } else {
                        for text in output_err.split("error: ").skip(1) {
                            let end = text.find('\n').unwrap_or(text.len());
                            let c_abi = "can't call C ABI function: ";
                            let unimplemented_intrinsic_s = "unimplemented intrinsic: ";
                            let unsupported_s = "miri does not support ";
                            let abi_s = "can't handle function with ";
                            let limit_s = "reached the configured maximum ";
                            if text.starts_with(c_abi) {
                                c_abi_fns.push(text[c_abi.len()..end].to_string());
                            } else if text.starts_with(unimplemented_intrinsic_s) {
                                unimplemented_intrinsic.push(text[unimplemented_intrinsic_s.len()..end].to_string());
                            } else if text.starts_with(unsupported_s) {
                                unsupported.push(text[unsupported_s.len()..end].to_string());
                            } else if text.starts_with(abi_s) {
                                abi.push(text[abi_s.len()..end].to_string());
                            } else if text.starts_with(limit_s) {
                                limits.push(text[limit_s.len()..end].to_string());
                            } else {
                                if text.find("aborting").is_none() {
                                    failed.push(text[..end].to_string());
                                }
                            }
                        }
                        writeln!(stderr.lock(), "FAILED with exit code {:?}", output.status.code()).unwrap();
                        writeln!(stderr.lock(), "stdout: \n {}", std::str::from_utf8(&output.stdout).unwrap()).unwrap();
                        writeln!(stderr.lock(), "stderr: \n {}", output_err).unwrap();
                    }
                }
                Err(e) => {
                    writeln!(stderr.lock(), "FAILED: {}", e).unwrap();
                    panic!("failed to execute miri");
                },
            }
        }
        let stderr = std::io::stderr();
        let mut stderr = stderr.lock();
        writeln!(stderr, "{} success, {} no mir, {} crate not found, {} failed, \
                          {} C fn, {} ABI, {} unsupported, {} intrinsic",
                          success, mir_not_found.len(), crate_not_found.len(), failed.len(),
                          c_abi_fns.len(), abi.len(), unsupported.len(), unimplemented_intrinsic.len()).unwrap();
        writeln!(stderr, "# The \"other reasons\" errors").unwrap();
        writeln!(stderr, "(sorted, deduplicated)").unwrap();
        print_vec(&mut stderr, failed);

        writeln!(stderr, "# can't call C ABI function").unwrap();
        print_vec(&mut stderr, c_abi_fns);

        writeln!(stderr, "# unsupported ABI").unwrap();
        print_vec(&mut stderr, abi);

        writeln!(stderr, "# unsupported").unwrap();
        print_vec(&mut stderr, unsupported);

        writeln!(stderr, "# unimplemented intrinsics").unwrap();
        print_vec(&mut stderr, unimplemented_intrinsic);

        writeln!(stderr, "# mir not found").unwrap();
        print_vec(&mut stderr, mir_not_found);

        writeln!(stderr, "# crate not found").unwrap();
        print_vec(&mut stderr, crate_not_found);

        panic!("ran miri on rustc test suite. Test failing for convenience");
    } else {
        run_pass();
        for_all_targets(&sysroot, |target| {
            miri_pass("tests/run-pass", &target, host);
        });
        compile_fail(&sysroot);
    }
}

fn print_vec<W: std::io::Write>(stderr: &mut W, v: Vec<String>) {
    writeln!(stderr, "```").unwrap();
    for (n, s) in vec_to_hist(v).into_iter().rev() {
        writeln!(stderr, "{:4} {}", n, s).unwrap();
    }
    writeln!(stderr, "```").unwrap();
}

fn vec_to_hist<T: PartialEq + Ord>(mut v: Vec<T>) -> Vec<(usize, T)> {
    v.sort();
    let mut v = v.into_iter();
    let mut result = Vec::new();
    let mut current = v.next();
    'outer: while let Some(current_val) = current {
        let mut n = 1;
        while let Some(next) = v.next() {
            if next == current_val {
                n += 1;
            } else {
                result.push((n, current_val));
                current = Some(next);
                continue 'outer;
            }
        }
        result.push((n, current_val));
        break;
    }
    result.sort();
    result
}
