#[cfg(not(target_os = "windows"))]
mod full {
    use std::env;
    use std::fs::File;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::path::Path;
    use std::process::{Command, Stdio};

    macro_rules! full_test {
        ($name:ident, $crate_name:expr, $old_version:expr, $new_version:expr) => {
            #[test]
            fn $name() {
                let mut success = true;

                let old_version = concat!($crate_name, ":", $old_version);
                let new_version = concat!($crate_name, ":", $new_version);

                let prog = concat!(r#"
    # wait for the actual output
    /^version bump/ {
        doprint = 1;
    }

    {
        # check the environ for filtering
        if (ENVIRON["RUST_LOG"] == "debug")
            doprint = 1;

        # skip compilation info
        if (!doprint)
            next;

        # sanitize paths
        gsub(/-->.*"#, $crate_name, r#"/, "--> "#, $crate_name, r#"", $0);
        print;
    }"#, $crate_name, $crate_name);
                let out_file = Path::new("tests/full_cases")
                    .join(concat!($crate_name, "-", $old_version, "-", $new_version));

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
                    let fd = stdin.as_raw_fd();
                    unsafe { (Stdio::from_raw_fd(fd), Stdio::from_raw_fd(fd)) }
                } else {
                    panic!("could not pipe to awk");
                };

                success &= Command::new("./target/debug/cargo-semver")
                    .args(&["-S", &old_version, "-C", &new_version])
                    .env("RUST_BACKTRACE", "full")
                    .stdin(Stdio::null())
                    .stdout(out_pipe)
                    .stderr(err_pipe)
                    .status()
                    .expect("could not run cargo semver")
                    .success();

                assert!(success, "cargo semver");

                success &= awk_child.wait().expect("could not wait for awk child").success();

                assert!(success, "awk");

                success &= Command::new("git")
                    .args(&["diff", "--exit-code", out_file])
                    .env("PAGER", "")
                    .status()
                    .expect("could not run git diff")
                    .success();

                assert!(success, "git");
            }
        }
    }

    full_test!(log, "log", "0.3.4", "0.3.8");
    full_test!(libc, "libc", "0.2.28", "0.2.31");
    // full_test!(mozjs, "mozjs", "0.2.0", "0.3.0");
    // full_test!(rand, "rand", "0.3.10", "0.3.16");
    // full_test!(serde_pre, "serde", "0.7.0", "1.0.0");
    // full_test!(serde_post, "serde", "1.0.0", "1.0.8");
}
