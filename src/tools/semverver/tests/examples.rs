mod features {
    use std::{
        env,
        fs::{read_to_string, File},
        io::Write,
        path::Path,
        process::{Command, Stdio},
        str,
    };

    fn test_example2(name: &str, path: &Path, expected_path: &Path, expected_result: bool) {
        let old_rlib = path.join("libold.rlib").to_str().unwrap().to_owned();
        let new_rlib = path.join("libnew.rlib").to_str().unwrap().to_owned();

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
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

        if let Ok(target_args) = &target_args {
            cmd.args(target_args);
        }

        if expected_path
            .to_str()
            .unwrap()
            .contains("stdout_api_guidelines")
        {
            cmd.env("RUST_SEMVER_API_GUIDELINES", "true");
        }

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

        let output = cmd.output().expect("could not run rust-semverver");

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
                .map(|line| {
                    // sanitize paths for reproducibility
                    match line.find("-->") {
                        Some(idx) => {
                            let (start, end) = line.split_at(idx);
                            match end.find(name) {
                                Some(idx) => format!("{}--> {}", start, end.split_at(idx).1),
                                None => line.to_string(),
                            }
                        }
                        None => line.to_string(),
                    }
                })
                .map(|l| {
                    if cfg!(target_os = "windows") {
                        l.replace('\\', "/")
                    } else {
                        l
                    }
                })
                .map(|l| l + "\n")
                .collect::<String>()
                .trim_end()
                .to_string()
        };

        if expected_output != new_output {
            eprintln!("rust-semverver failed to produce the expected result");

            let new_path = Path::new(&env::var("OUT_DIR").unwrap()).join(name);
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
            "rust-semverver returned an unexpected exit status"
        );

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
                test_example2(stringify!($name), &path, &path.join("stdout"), $result);

                if path.join("stdout_api_guidelines").exists() {
                    eprintln!("api-guidelines");
                    test_example2(stringify!($name), &path, &path.join("stdout_api_guidelines"), $result);
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
