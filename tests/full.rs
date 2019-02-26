mod full {
    use std::{
        env,
        fs::File,
        path::{Path, PathBuf},
        process::{Command, Stdio},
    };

    fn test_full(
        crate_name: &str,
        old_version: &str,
        new_version: &str,
        expected_result: bool
    ) {
        let prog = format!(
            r#"
    # wait for the actual output
    /^version bump/ {{
        doprint = 1;
    }}

    {{
        # check the environ for filtering
        if (ENVIRON["RUST_LOG"] == "debug")
            doprint = 1;

        # skip compilation info
        if (!doprint)
            next;

        # sanitize paths
        gsub(/-->.*{crate_name}/, "--> {crate_name}", $0);
        print;
    }}{crate_name}{crate_name}"#,
            crate_name = crate_name
        );

        let out_file = Path::new("tests/full_cases")
            .join(format!("{}-{}-{}", crate_name, old_version, new_version));

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

        let out_file: PathBuf = format!("{}.{}", out_file.display(), file_ext).into();
        assert!(out_file.exists());

        if let Some(path) = env::var_os("PATH") {
            let mut paths = env::split_paths(&path).collect::<Vec<_>>();
            let current_dir = env::current_dir().expect("could not determine current dir");
            paths.insert(0, current_dir.join("target/debug"));
            let new_path = env::join_paths(paths).unwrap();
            env::set_var("PATH", &new_path);
        } else {
            eprintln!("no path!");
        }

        let stdout = File::create(&out_file).expect("could not create `stdout` file");
        let out_file = out_file.to_str().unwrap();

        let mut awk_child = Command::new("awk")
            .arg(&prog)
            .stdin(Stdio::piped())
            .stdout(stdout)
            .spawn()
            .expect("could not run awk");

        let (err_pipe, out_pipe) = if let Some(ref stdin) = awk_child.stdin {
            #[cfg(unix)]
            {
                use std::os::unix::io::{AsRawFd, FromRawFd};
                let fd = stdin.as_raw_fd();
                unsafe { (Stdio::from_raw_fd(fd), Stdio::from_raw_fd(fd)) }
            }
            #[cfg(windows)]
            {
                use std::os::windows::io::{AsRawHandle, FromRawHandle};
                let fd = stdin.as_raw_handle();
                unsafe { (Stdio::from_raw_handle(fd), Stdio::from_raw_handle(fd)) }
            }
        } else {
            panic!("could not pipe to awk");
        };

        let old_version = format!("{}:{}", crate_name, old_version);
        let new_version = format!("{}:{}", crate_name, new_version);

        let cargo_semver_result = {
            let mut cmd = Command::new("./target/debug/cargo-semver");
            cmd.args(&["-S", &old_version, "-C", &new_version])
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null())
                .stdout(out_pipe)
                .stderr(err_pipe);

            if let Ok(target) = std::env::var("TEST_TARGET") {
                cmd.args(&["--target", &target]);
            }

            cmd.status().expect("could not run cargo semver").success()
        };

        assert_eq!(
            cargo_semver_result, expected_result,
            "cargo semver returned an unexpected exit status"
        );

        let awk_result = awk_child
            .wait()
            .expect("could not wait for awk child")
            .success();

        assert!(awk_result, "awk");

        let git_result = Command::new("git")
            .args(&["diff", "--ignore-space-at-eol", "--exit-code", out_file])
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
    full_test!(libc, "libc", "0.2.28", "0.2.31", false);
    // full_test!(mozjs, "mozjs", "0.2.0", "0.3.0");
    // full_test!(rand, "rand", "0.3.10", "0.3.16");
    // full_test!(serde_pre, "serde", "0.7.0", "1.0.0");
    // full_test!(serde_post, "serde", "1.0.0", "1.0.8");
}
