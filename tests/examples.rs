use std::env;
use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::Path;
use std::process::{Command, Stdio};

macro_rules! test {
    ($name:ident) => {
        #[test]
        fn $name() {
            let mut success = true;

            let current_dir = env::current_dir().expect("could not determine current dir");
            let subst = format!("s#{}#$REPO_PATH#g", current_dir.to_str().unwrap());
            let path = Path::new("tests/cases").join(stringify!($name));

            let out_file = path.join("stdout");

            let old_rlib = path.join("libold.rlib").to_str().unwrap().to_owned();
            let new_rlib = path.join("libnew.rlib").to_str().unwrap().to_owned();

            {
                let stdout = File::create(&out_file).expect("could not create `stdout` file");

                success &= Command::new("rustc")
                    .args(&["--crate-type=lib", "-o", &old_rlib])
                    .arg(path.join("old.rs"))
                    .env("RUST_BACKTRACE", "full")
                    .stdin(Stdio::null())
                    .status()
                    .expect("could not run rustc")
                    .success();

                assert!(success, "couldn't compile old");

                success &= Command::new("rustc")
                    .args(&["--crate-type=lib", "-o", &new_rlib])
                    .arg(path.join("new.rs"))
                    .env("RUST_BACKTRACE", "full")
                    .stdin(Stdio::null())
                    .status()
                    .expect("could not run rustc")
                    .success();

                assert!(success, "couldn't compile new");

                let mut sed_child = Command::new("sed")
                    .arg(&subst)
                    .stdin(Stdio::piped())
                    .stdout(stdout)
                    .spawn()
                    .expect("could not run sed");

                let (err_pipe, out_pipe) = if let Some(ref stdin) = sed_child.stdin {
                    let fd = stdin.as_raw_fd();
                    unsafe { (Stdio::from_raw_fd(fd), Stdio::from_raw_fd(fd)) }
                } else {
                    panic!("could not pipe to sed");
                };

                success &= Command::new("./target/debug/rust-semverver")
                    .args(&["--crate-type=lib", "-Zverbose",
                            "--extern", &format!("old={}", old_rlib),
                            "--extern", &format!("new={}", new_rlib),
                            "tests/helper/test.rs"])
                    .env("RUST_BACKTRACE", "full")
                    .env("RUST_SEMVER_CRATE_VERSION", "1.0.0")
                    .stdin(Stdio::null())
                    .stdout(out_pipe)
                    .stderr(err_pipe)
                    .status()
                    .expect("could not run rust-semverver")
                    .success();

                assert!(success, "rust-semverver");

                success &= sed_child.wait().expect("could not wait for sed child").success();
            }

            assert!(success, "sed");

            eprintln!("path: {}", out_file.to_str().unwrap());
            success &= Command::new("git")
                .args(&["diff", "--quiet", out_file.to_str().unwrap()])
                .status()
                .expect("could not run git diff")
                .success();

            assert!(success, "git");

            Command::new("rm")
                .args(&[&old_rlib, &new_rlib])
                .status()
                .expect("could not run rm");
        }
    }
}

macro_rules! full_test {
    ($name:ident, $old_version:expr, $new_version:expr) => {
        #[test]
        fn $name() {
            let mut success = true;

            let old_version = concat!(stringify!($name), "-", $old_version);
            let new_version = concat!(stringify!($name), "-", $new_version);

            let current_dir = env::current_dir().expect("could not determine current dir");
            let subst = format!("s#{}#$REPO_PATH#g", current_dir.to_str().unwrap());
            let out_file = Path::new("tests/full_cases")
                .join(concat!(stringify!($name), "-", $old_version, "-", $new_version));

            if let Some(path) = env::var_os("PATH") {
                let mut paths = env::split_paths(&path).collect::<Vec<_>>();
                paths.push(current_dir.join("target/debug"));
                let new_path = env::join_paths(paths).unwrap();
                env::set_var("PATH", &new_path);
            }

            let stdout = File::create(&out_file).expect("could not create `stdout` file");
            let out_file = out_file.to_str().unwrap();

            let mut sed_child = Command::new("sed")
                .arg(&subst)
                .stdin(Stdio::piped())
                .stdout(stdout)
                .spawn()
                .expect("could not run sed");

            let (err_pipe, out_pipe) = if let Some(ref stdin) = sed_child.stdin {
                let fd = stdin.as_raw_fd();
                unsafe { (Stdio::from_raw_fd(fd), Stdio::from_raw_fd(fd)) }
            } else {
                panic!("could not pipe to sed");
            };

            success &= Command::new("./target/debug/cargo-semver")
                .args(&["semver", "-S", &old_version, "-C", &new_version])
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null())
                .stdout(out_pipe)
                .stderr(err_pipe)
                .status()
                .expect("could not run cargo-semver")
                .success();

            assert!(success, "cargo-semver");

            success &= sed_child.wait().expect("could not wait for sed child").success();

            assert!(success, "sed");

            eprintln!("path: {}", out_file);
            success &= Command::new("git")
                .args(&["diff", "--quiet", out_file])
                .status()
                .expect("could not run git diff")
                .success();

            assert!(success, "git");
        }
    }
}

test!(addition);
test!(addition_path);
test!(addition_use);
test!(bounds);
test!(consts);
test!(enums);
test!(func);
test!(infer);
test!(infer_regress);
test!(inherent_impls);
test!(kind_change);
test!(macros);
test!(max_priv);
test!(mix);
test!(pathologic_paths);
test!(pub_use);
test!(regions);
test!(removal);
test!(removal_path);
test!(removal_use);
test!(structs);
test!(swap);
test!(traits);
test!(trait_impls);
test!(ty_alias);

full_test!(log, "0.3.4", "0.3.8");
full_test!(serde, "0.7.0", "1.0.0");
// full_test!(serde, "1.0.0", "1.0.8");
