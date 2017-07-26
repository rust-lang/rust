use std::env;
use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::Path;
use std::process::{Command, Stdio};

macro_rules! test {
    ($name:ident, $path:expr) => {
        #[test]
        fn $name() {
            let mut success = true;

            let current_dir = env::current_dir().expect("could not determine current dir");
            let subst = format!("s#{}#$REPO_PATH#g", current_dir.to_str().unwrap());
            let path = Path::new($path);

            let out_file = path.join("stdout");

            let old_rlib = path.join("libold.rlib");
            let new_rlib = path.join("libnew.rlib");

            {
                let stdout = File::create(&out_file).expect("could not create `stdout` file");

                success &= Command::new("rustc")
                    .args(&["--crate-type=lib", "-o", old_rlib.to_str().unwrap()])
                    .arg(path.join("old.rs"))
                    .env("RUST_BACKTRACE", "full")
                    .stdin(Stdio::null())
                    .status()
                    .expect("could not run rustc")
                    .success();

                assert!(success, "couldn't compile old");

                success &= Command::new("rustc")
                    .args(&["--crate-type=lib", "-o", new_rlib.to_str().unwrap()])
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
                            "--extern", &format!("old={}", old_rlib.to_str().unwrap()),
                            "--extern", &format!("new={}", new_rlib.to_str().unwrap()),
                            "tests/helper/test.rs"])
                    .env("RUST_LOG", "debug")
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
                .args(&[old_rlib.to_str().unwrap(), new_rlib.to_str().unwrap()])
                .status()
                .expect("could not run rm");
        }
    }
}

test!(addition, "tests/cases/addition");
test!(addition_path, "tests/cases/addition_path");
test!(addition_use, "tests/cases/addition_use");
test!(bounds, "tests/cases/bounds");
test!(enums, "tests/cases/enums");
test!(func, "tests/cases/func");
test!(infer, "tests/cases/infer");
test!(kind_change, "tests/cases/kind_change");
test!(macros, "tests/cases/macros");
test!(mix, "tests/cases/mix");
test!(pathologic_paths, "tests/cases/pathologic_paths");
test!(pub_use, "tests/cases/pub_use");
test!(regions, "tests/cases/regions");
test!(removal, "tests/cases/removal");
test!(removal_path, "tests/cases/removal_path");
test!(removal_use, "tests/cases/removal_use");
test!(structs, "tests/cases/structs");
test!(swap, "tests/cases/swap");
test!(traits, "tests/cases/traits");
test!(ty_alias, "tests/cases/ty_alias");
