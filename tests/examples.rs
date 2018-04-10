mod features {
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
                    .args(&["diff", "--exit-code", out_file.to_str().unwrap()])
                    .env("PAGER", "")
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

    test!(addition);
    test!(addition_path);
    test!(addition_use);
    test!(bounds);
    test!(circular);
    test!(consts);
    test!(enums);
    test!(func);
    test!(func_local_items);
    test!(infer);
    test!(infer_regress);
    test!(inherent_impls);
    test!(issue_34);
    test!(issue_50);
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
    test!(sealed_traits);
    test!(structs);
    test!(swap);
    test!(traits);
    test!(trait_impls);
    test!(trait_objects);
    test!(ty_alias);
}
