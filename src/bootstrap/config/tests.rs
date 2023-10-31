use crate::config::TomlConfig;

use super::{Config, Flags};
use clap::CommandFactory;
use serde::Deserialize;
use std::{env, path::Path};

fn parse(config: &str) -> Config {
    Config::parse_inner(&["check".to_owned(), "--config=/does/not/exist".to_owned()], |&_| {
        toml::from_str(config).unwrap()
    })
}

#[test]
fn download_ci_llvm() {
    if crate::llvm::is_ci_llvm_modified(&parse("")) {
        eprintln!("Detected LLVM as non-available: running in CI and modified LLVM in this change");
        return;
    }

    let parse_llvm = |s| parse(s).llvm_from_ci;
    let if_available = parse_llvm("llvm.download-ci-llvm = \"if-available\"");

    assert!(parse_llvm("llvm.download-ci-llvm = true"));
    assert!(!parse_llvm("llvm.download-ci-llvm = false"));
    assert_eq!(parse_llvm(""), if_available);
    assert_eq!(parse_llvm("rust.channel = \"dev\""), if_available);
    assert!(!parse_llvm("rust.channel = \"stable\""));
    assert!(parse_llvm("build.build = \"x86_64-unknown-linux-gnu\""));
    assert!(parse_llvm(
        "llvm.assertions = true \r\n build.build = \"x86_64-unknown-linux-gnu\" \r\n llvm.download-ci-llvm = \"if-available\""
    ));
    assert!(!parse_llvm(
        "llvm.assertions = true \r\n build.build = \"aarch64-apple-darwin\" \r\n llvm.download-ci-llvm = \"if-available\""
    ));
}

// FIXME(ozkanonur): extend scope of the test
// refs:
//   - https://github.com/rust-lang/rust/issues/109120
//   - https://github.com/rust-lang/rust/pull/109162#issuecomment-1496782487
#[test]
fn detect_src_and_out() {
    fn test(cfg: Config, build_dir: Option<&str>) {
        // This will bring absolute form of `src/bootstrap` path
        let current_dir = std::env::current_dir().unwrap();

        // get `src` by moving into project root path
        let expected_src = current_dir.ancestors().nth(2).unwrap();
        assert_eq!(&cfg.src, expected_src);

        // Sanity check for `src`
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let expected_src = manifest_dir.ancestors().nth(2).unwrap();
        assert_eq!(&cfg.src, expected_src);

        // test if build-dir was manually given in config.toml
        if let Some(custom_build_dir) = build_dir {
            assert_eq!(&cfg.out, Path::new(custom_build_dir));
        }
        // test the native bootstrap way
        else {
            // This should bring output path of bootstrap in absolute form
            let cargo_target_dir = env::var_os("CARGO_TARGET_DIR").expect(
                "CARGO_TARGET_DIR must been provided for the test environment from bootstrap",
            );

            // Move to `build` from `build/bootstrap`
            let expected_out = Path::new(&cargo_target_dir).parent().unwrap();
            assert_eq!(&cfg.out, expected_out);

            let args: Vec<String> = env::args().collect();

            // Another test for `out` as a sanity check
            //
            // This will bring something similar to:
            //     `{build-dir}/bootstrap/debug/deps/bootstrap-c7ee91d5661e2804`
            // `{build-dir}` can be anywhere, not just in the rust project directory.
            let dep = Path::new(args.first().unwrap());
            let expected_out = dep.ancestors().nth(4).unwrap();

            assert_eq!(&cfg.out, expected_out);
        }
    }

    test(parse(""), None);

    {
        let build_dir = if cfg!(windows) { Some("C:\\tmp") } else { Some("/tmp") };
        test(parse("build.build-dir = \"/tmp\""), build_dir);
    }
}

#[test]
fn clap_verify() {
    Flags::command().debug_assert();
}

#[test]
fn override_toml() {
    let config = Config::parse_inner(
        &[
            "check".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--set=changelog-seen=1".to_owned(),
            "--set=rust.lto=fat".to_owned(),
            "--set=rust.deny-warnings=false".to_owned(),
            "--set=build.gdb=\"bar\"".to_owned(),
            "--set=build.tools=[\"cargo\"]".to_owned(),
            "--set=llvm.build-config={\"foo\" = \"bar\"}".to_owned(),
        ],
        |&_| {
            toml::from_str(
                r#"
changelog-seen = 0
[rust]
lto = "off"
deny-warnings = true

[build]
gdb = "foo"
tools = []

[llvm]
download-ci-llvm = false
build-config = {}
                "#,
            )
            .unwrap()
        },
    );
    assert_eq!(config.changelog_seen, Some(1), "setting top-level value");
    assert_eq!(
        config.rust_lto,
        crate::config::RustcLto::Fat,
        "setting string value without quotes"
    );
    assert_eq!(config.gdb, Some("bar".into()), "setting string value with quotes");
    assert_eq!(config.deny_warnings, false, "setting boolean value");
    assert_eq!(
        config.tools,
        Some(["cargo".to_string()].into_iter().collect()),
        "setting list value"
    );
    assert_eq!(
        config.llvm_build_config,
        [("foo".to_string(), "bar".to_string())].into_iter().collect(),
        "setting dictionary value"
    );
}

#[test]
#[should_panic]
fn override_toml_duplicate() {
    Config::parse_inner(
        &[
            "check".to_owned(),
            "--config=/does/not/exist".to_owned(),
            "--set=changelog-seen=1".to_owned(),
            "--set=changelog-seen=2".to_owned(),
        ],
        |&_| toml::from_str("changelog-seen = 0").unwrap(),
    );
}

#[test]
fn profile_user_dist() {
    fn get_toml(file: &Path) -> TomlConfig {
        let contents = if file.ends_with("config.toml") {
            "profile = \"user\"".to_owned()
        } else {
            assert!(file.ends_with("config.dist.toml"));
            std::fs::read_to_string(dbg!(file)).unwrap()
        };
        toml::from_str(&contents)
            .and_then(|table: toml::Value| TomlConfig::deserialize(table))
            .unwrap()
    }
    Config::parse_inner(&["check".to_owned()], get_toml);
}

#[test]
fn rust_optimize() {
    assert_eq!(parse("").rust_optimize.is_release(), true);
    assert_eq!(parse("rust.optimize = false").rust_optimize.is_release(), false);
    assert_eq!(parse("rust.optimize = true").rust_optimize.is_release(), true);
    assert_eq!(parse("rust.optimize = 0").rust_optimize.is_release(), false);
    assert_eq!(parse("rust.optimize = 1").rust_optimize.is_release(), true);
    assert_eq!(parse("rust.optimize = 1").rust_optimize.get_opt_level(), Some("1".to_string()));
    assert_eq!(parse("rust.optimize = \"s\"").rust_optimize.is_release(), true);
    assert_eq!(parse("rust.optimize = \"s\"").rust_optimize.get_opt_level(), Some("s".to_string()));
}

#[test]
#[should_panic]
fn invalid_rust_optimize() {
    parse("rust.optimize = \"a\"");
}
