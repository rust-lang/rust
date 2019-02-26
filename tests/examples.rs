mod features {
    use std::{
        env, fs, io,
        path::Path,
        process::{Command, Stdio},
    };

    fn test_example(path: &Path, out_file: &Path, expected_result: bool) {
        let old_rlib = path.join("libold.rlib").to_str().unwrap().to_owned();
        let new_rlib = path.join("libnew.rlib").to_str().unwrap().to_owned();

        {
            let stdout = fs::File::create(&out_file).expect("could not create `stdout` file");
            let stderr = stdout
                .try_clone()
                .expect("could not create `stderr` file by cloning `stdout`");

            let target_args = std::env::var("TEST_TARGET").map(|t| ["--target".to_string(), t]);

            let mut cmd = Command::new("rustc");
            cmd.args(&["--crate-type=lib", "-o", &old_rlib])
                .arg(path.join("old.rs"))
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null());

            if let Ok(target_args) = &target_args {
                cmd.args(target_args);
            }

            let rustc_old_result = cmd.status().expect("could not run rustc on old").success();
            assert!(rustc_old_result, "couldn't compile old");

            let mut cmd = Command::new("rustc");
            cmd.args(&["--crate-type=lib", "-o", &new_rlib])
                .arg(path.join("new.rs"))
                .env("RUST_BACKTRACE", "full")
                .stdin(Stdio::null());

            if let Ok(target_args) = &target_args {
                cmd.args(target_args);
            }

            let rustc_new_result = cmd.status().expect("could not run rustc on new").success();
            assert!(rustc_new_result, "couldn't compile new");

            let mut cmd = Command::new(
                Path::new(".")
                    .join("target")
                    .join("debug")
                    .join("rust-semverver")
                    .to_str()
                    .unwrap(),
            );
            cmd.args(&[
                "--crate-type=lib",
                "-Zverbose",
                "--extern",
                &format!("old={}", old_rlib),
                "--extern",
                &format!("new={}", new_rlib),
                Path::new("tests")
                    .join("helper")
                    .join("test.rs")
                    .to_str()
                    .unwrap(),
            ])
            .env("RUST_BACKTRACE", "full")
            .env("RUST_SEMVER_CRATE_VERSION", "1.0.0")
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));

            if let Ok(target_args) = &target_args {
                cmd.args(target_args);
            }

            if out_file.to_str().unwrap().contains("stdout_api_guidelines") {
                cmd.env("RUST_SEMVER_API_GUIDELINES", "true");
            }

            let rustsemverver_result = cmd
                .status()
                .expect("could not run rust-semverver")
                .success();
            assert_eq!(
                rustsemverver_result, expected_result,
                "rust-semverver returned an unexpected exit status"
            );

            {
                // replace root path with with $REPO_PATH
                use self::io::{Read, Write};
                let current_dir = env::current_dir().expect("could not determine current dir");
                let mut contents = {
                    let mut f = fs::File::open(&out_file).expect("file not found");
                    let mut contents = String::new();
                    f.read_to_string(&mut contents)
                        .expect("something went wrong reading the file");
                    contents
                };

                contents = contents.replace(current_dir.to_str().unwrap(), "$REPO_PATH");

                if cfg!(target_os = "windows") {
                    let mut lines = Vec::new();

                    for line in contents.lines() {
                        if line.contains("$REPO_PATH") {
                            lines.push(line.replace('\\', "/"));
                        } else {
                            lines.push(line.to_owned());
                        }
                    }
                    lines.push(String::new());
                    contents = lines.join("\r\n");
                }

                let mut file = fs::File::create(&out_file).expect("cannot create file");
                file.write_all(contents.as_bytes())
                    .expect("cannot write to file");
            }
        }

        let git_result = Command::new("git")
            .args(&[
                "diff",
                "--ignore-space-at-eol",
                "--exit-code",
                out_file.to_str().unwrap(),
            ])
            .env("PAGER", "")
            .status()
            .expect("could not run git diff")
            .success();

        assert!(git_result, "git reports unexpected diff");

        Command::new("rm")
            .args(&[&old_rlib, &new_rlib])
            .status()
            .expect("could not run rm");
    }

    macro_rules! test {
        ($name:ident => $result:literal) => {
            #[test]
            fn $name() {
                let path = Path::new("tests").join("cases").join(stringify!($name));
                test_example(&path, &path.join("stdout"), $result);

                if path.join("stdout_api_guidelines").exists() {
                    eprintln!("api-guidelines");
                    test_example(&path, &path.join("stdout_api_guidelines"), $result);
                }
            }
        };
        ($($name:ident => $result:literal),*) => {
            $(test!($name => $result);)*
        };
        ($($name:ident => $result:literal,)*) => {
            $(test!($name => $result);)*
        };
    }

    test! {
        addition => true,
        addition_path => true,
        addition_use => false,
        bounds => false,
        circular => true,
        consts => false,
        enums => false,
        func => false,
        func_local_items => true,
        infer => true,
        infer_regress => false,
        inherent_impls => false,
        issue_34 => true,
        issue_50 => true,
        kind_change => false,
        macros => false,
        max_priv => true,
        mix => false,
        pathologic_paths => true,
        pub_use => true,
        regions => false,
        removal => false,
        removal_path => false,
        removal_use => false,
        sealed_traits => true,
        structs => false,
        swap => true,
        traits => false,
        trait_impls => false,
        trait_objects => true,
        ty_alias => false,
    }
}
