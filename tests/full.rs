mod full {
    use log::{log_enabled, Level};
    use std::{
        env,
        fs::File,
        io::{BufRead, Write},
        path::Path,
        process::{Command, Stdio},
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

        assert_eq!(
            output.status.success(),
            expected_result,
            "cargo-semver returned unexpected exit status {}",
            output.status
        );

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

        let filename = Path::new("tests/full_cases").join(format!(
            "{}-{}-{}.{}",
            crate_name, old_version, new_version, file_ext
        ));

        assert!(
            filename.exists(),
            "file `{}` does not exist",
            filename.display()
        );

        let mut file = File::create(&filename).expect("could not create output file");

        for line in output
            .stdout
            .lines()
            .chain(output.stderr.lines())
            .map(|r| r.expect("could not read line from cargo-semver output"))
            .skip_while(|line|
                // skip everything before the first important bit of info
                !line.starts_with("version bump") &&
                // ...unless debugging is enabled
                !log_enabled!(Level::Debug))
        {
            // sanitize paths for reproducibility
            let output = match line.find("-->") {
                Some(idx) => {
                    let (start, end) = line.split_at(idx);
                    match end.find(crate_name) {
                        Some(idx) => format!("{}--> {}", start, end.split_at(idx).1),
                        None => line,
                    }
                }
                None => line,
            };
            writeln!(file, "{}", output).expect("error writing to output file");
        }

        let git_result = Command::new("git")
            .args(&[
                "diff",
                "--ignore-space-at-eol",
                "--exit-code",
                filename.to_str().unwrap(),
            ])
            .env("PAGER", "")
            .status()
            .expect("could not run git diff")
            .success();

        assert!(git_result, "git reports unexpected diff");
    }

    macro_rules! full_test {
        ($name:ident, $crate_name:expr,
         $old_version:expr, $new_version:expr,
         $result:literal) => {
            #[test]
            fn $name() {
                test_full($crate_name, $old_version, $new_version, $result);
            }
        };
    }

    full_test!(log, "log", "0.3.4", "0.3.8", true);
    full_test!(libc0, "libc", "0.2.28", "0.2.31", false);
    full_test!(libc1, "libc", "0.2.47", "0.2.48", true);
    // full_test!(mozjs, "mozjs", "0.2.0", "0.3.0");
    // full_test!(rand, "rand", "0.3.10", "0.3.16");
    // full_test!(serde_pre, "serde", "0.7.0", "1.0.0");
    // full_test!(serde_post, "serde", "1.0.0", "1.0.8");
}
