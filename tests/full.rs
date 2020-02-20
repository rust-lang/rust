mod full {
    use log::{log_enabled, Level};
    use std::{
        env,
        fs::{read_to_string, File},
        io::Write,
        path::{Path, PathBuf},
        process::{Command, Stdio},
        str,
    };

    fn test_full(crate_name: &str, old_version: &str, new_version: &str, expected_result: bool) {
        // Add target dir to PATH so cargo-semver will call the right rust-semverver
        if let Some(path) = env::var_os("PATH") {
            let mut paths = env::split_paths(&path).collect::<Vec<_>>();
            let current_dir = env::current_dir().expect("could not determine current dir");
            paths.insert(0, current_dir.join("target/debug"));
            let new_path = env::join_paths(paths).unwrap();
            env::set_var("PATH", &new_path);
        } else {
            eprintln!("no path!");
        }

        let mut cmd = Command::new("./target/debug/cargo-semver");
        cmd.args(&[
            "-S",
            &format!("{}:{}", crate_name, old_version),
            "-C",
            &format!("{}:{}", crate_name, new_version),
            "-q",
        ])
        .env("RUST_BACKTRACE", "full")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

        if let Ok(target) = std::env::var("TEST_TARGET") {
            cmd.args(&["--target", &target]);
        }

        let output = cmd.output().expect("could not run cargo semver");

        // Choose solution depending on the platform
        let file_ext = if cfg!(target_os = "macos") {
            "osx"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(all(target_os = "windows", target_env = "msvc")) {
            "windows_msvc"
        } else {
            eprintln!("full tests are not available in this target");
            return;
        };

        let filename = format!(
            "{}-{}-{}.{}",
            crate_name, old_version, new_version, file_ext
        );

        let expected_path = {
            let mut buf = PathBuf::new();

            buf.push("tests");
            buf.push("full_cases");
            buf.push(&filename);

            buf
        };

        let expected_output = read_to_string(&expected_path)
            .expect(&format!(
                "could not read expected output from file {}",
                expected_path.display()
            ))
            .lines()
            .map(|l| l.trim_end())
            .map(|l| l.to_string() + "\n")
            .collect::<String>()
            .trim_end()
            .to_string();

        let new_output = {
            let stdout: &str = str::from_utf8(&output.stdout)
                .expect("could not read line from rust-semverver output")
                .trim_end();
            let stderr: &str = str::from_utf8(&output.stderr)
                .expect("could not read line from rust-semverver output")
                .trim_end();

            stdout
                .lines()
                .chain(stderr.lines())
                .map(|l| l.trim_end())
                .skip_while(|line|
                    // skip everything before the first important bit of info
                    !line.starts_with("version bump") &&
                        // ...unless debugging is enabled
                        !log_enabled!(Level::Debug))
                .map(|line| {
                    // sanitize paths for reproducibility
                    match line.find("-->") {
                        Some(idx) => {
                            let (start, end) = line.split_at(idx);
                            match end.find(crate_name) {
                                Some(idx) => format!("{}--> {}", start, end.split_at(idx).1),
                                None => line.to_string(),
                            }
                        }
                        None => line.to_string(),
                    }
                })
                .map(|l| l + "\n")
                .collect::<String>()
                .trim_end()
                .to_string()
        };

        if expected_output != new_output {
            eprintln!("cargo-semver failed to produce the expected output");

            let new_path = Path::new(&env::var("OUT_DIR").unwrap()).join(filename);
            let mut new_file = File::create(&new_path).unwrap();
            new_file.write_all(new_output.as_bytes()).unwrap();

            match std::env::var_os("CI") {
                None => {
                    eprintln!(
                        "For details, try this command:\n\n    diff {} {}\n\n",
                        expected_path.display(),
                        new_path.display()
                    );
                }
                Some(_) => {
                    eprintln!("=== Expected output ===");
                    eprintln!("{}", expected_output);
                    eprintln!("=== End of expected output ===");
                    eprintln!("=== Actual output ===");
                    eprintln!("{}", new_output);
                    eprintln!("=== End of actual output ===");
                }
            };

            panic!("unexpected output diff");
        }

        assert_eq!(
            output.status.success(),
            expected_result,
            "cargo-semver returned unexpected exit status {}",
            output.status
        );
    }

    macro_rules! full_test {
        ($name:ident, $crate_name:expr,
         $old_version:expr, $new_version:expr,
         $result:expr) => {
            #[test]
            fn $name() {
                test_full($crate_name, $old_version, $new_version, $result);
            }
        };
    }

    full_test!(log, "log", "0.3.4", "0.3.8", true);
    // the libc API on windows did *not* change between these versions
    full_test!(libc0, "libc", "0.2.28", "0.2.31", cfg!(windows));
    full_test!(libc1, "libc", "0.2.47", "0.2.48", true);
    full_test!(rmpv, "rmpv", "0.4.0", "0.4.1", false);
    // full_test!(mozjs, "mozjs", "0.2.0", "0.3.0");
    // full_test!(rand, "rand", "0.3.10", "0.3.16");
    // full_test!(serde_pre, "serde", "0.7.0", "1.0.0");
    // full_test!(serde_post, "serde", "1.0.0", "1.0.8");
}
