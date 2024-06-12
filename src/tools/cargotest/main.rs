use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

struct Test {
    repo: &'static str,
    name: &'static str,
    sha: &'static str,
    lock: Option<&'static str>,
    packages: &'static [&'static str],
    features: Option<&'static [&'static str]>,
    manifest_path: Option<&'static str>,
    /// `filters` are passed to libtest (i.e., after a `--` in the `cargo test` invocation).
    filters: &'static [&'static str],
}

const TEST_REPOS: &[Test] = &[
    Test {
        name: "iron",
        repo: "https://github.com/iron/iron",
        sha: "cf056ea5e8052c1feea6141e40ab0306715a2c33",
        lock: None,
        packages: &[],
        features: None,
        manifest_path: None,
        filters: &[],
    },
    Test {
        name: "ripgrep",
        repo: "https://github.com/BurntSushi/ripgrep",
        sha: "ced5b92aa93eb47e892bd2fd26ab454008721730",
        lock: None,
        packages: &[],
        features: None,
        manifest_path: None,
        filters: &[],
    },
    Test {
        name: "tokei",
        repo: "https://github.com/XAMPPRocky/tokei",
        sha: "fdf3f8cb279a7aeac0696c87e5d8b0cd946e4f9e",
        lock: None,
        packages: &[],
        features: None,
        manifest_path: None,
        filters: &[],
    },
    Test {
        name: "xsv",
        repo: "https://github.com/BurntSushi/xsv",
        sha: "3de6c04269a7d315f7e9864b9013451cd9580a08",
        lock: None,
        packages: &[],
        features: None,
        manifest_path: None,
        // Many tests here use quickcheck and some of them can fail randomly, so only run deterministic tests.
        filters: &[
            "test_flatten::",
            "test_fmt::",
            "test_headers::",
            "test_index::",
            "test_join::",
            "test_partition::",
            "test_search::",
            "test_select::",
            "test_slice::",
            "test_split::",
            "test_stats::",
            "test_table::",
        ],
    },
    Test {
        name: "servo",
        repo: "https://github.com/servo/servo",
        sha: "785a344e32db58d4e631fd3cae17fd1f29a721ab",
        lock: None,
        // Only test Stylo a.k.a. Quantum CSS, the parts of Servo going into Firefox.
        // This takes much less time to build than all of Servo and supports stable Rust.
        packages: &["selectors"],
        features: None,
        manifest_path: None,
        filters: &[],
    },
    Test {
        name: "diesel",
        repo: "https://github.com/diesel-rs/diesel",
        sha: "91493fe47175076f330ce5fc518f0196c0476f56",
        lock: None,
        packages: &[],
        // Test the embedded sqlite variant of diesel
        // This does not require any dependency to be present,
        // sqlite will be compiled as part of the build process
        features: Some(&["sqlite", "libsqlite3-sys/bundled"]),
        // We are only interested in testing diesel itself
        // not any other crate present in the diesel workspace
        // (This is required to set the feature flags above)
        manifest_path: Some("diesel/Cargo.toml"),
        filters: &[],
    },
];

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let cargo = &args[1];
    let out_dir = Path::new(&args[2]);
    let cargo = &Path::new(cargo);

    for test in TEST_REPOS.iter().rev() {
        if args[3..].is_empty() || args[3..].iter().any(|s| s.contains(test.name)) {
            test_repo(cargo, out_dir, test);
        }
    }
}

fn test_repo(cargo: &Path, out_dir: &Path, test: &Test) {
    println!("testing {}", test.repo);
    let dir = clone_repo(test, out_dir);
    if let Some(lockfile) = test.lock {
        fs::write(&dir.join("Cargo.lock"), lockfile).unwrap();
    }
    if !run_cargo_test(cargo, &dir, test.packages, test.features, test.manifest_path, test.filters)
    {
        panic!("tests failed for {}", test.repo);
    }
}

fn clone_repo(test: &Test, out_dir: &Path) -> PathBuf {
    let out_dir = out_dir.join(test.name);

    if !out_dir.join(".git").is_dir() {
        let status = Command::new("git").arg("init").arg(&out_dir).status().unwrap();
        assert!(status.success());
    }

    // Try progressively deeper fetch depths to find the commit
    let mut found = false;
    for depth in &[0, 1, 10, 100, 1000, 100000] {
        if *depth > 0 {
            let status = Command::new("git")
                .arg("fetch")
                .arg(test.repo)
                .arg(test.sha)
                .arg(&format!("--depth={}", depth))
                .current_dir(&out_dir)
                .status()
                .unwrap();
            assert!(status.success());
        }

        let status = Command::new("git")
            .arg("reset")
            .arg(test.sha)
            .arg("--hard")
            .current_dir(&out_dir)
            .status()
            .unwrap();

        if status.success() {
            found = true;
            break;
        }
    }

    if !found {
        panic!("unable to find commit {}", test.sha)
    }
    let status =
        Command::new("git").arg("clean").arg("-fdx").current_dir(&out_dir).status().unwrap();
    assert!(status.success());

    out_dir
}

fn run_cargo_test(
    cargo_path: &Path,
    crate_path: &Path,
    packages: &[&str],
    features: Option<&[&str]>,
    manifest_path: Option<&str>,
    filters: &[&str],
) -> bool {
    let mut command = Command::new(cargo_path);
    command.arg("test");

    if let Some(path) = manifest_path {
        command.arg(format!("--manifest-path={}", path));
    }

    if let Some(features) = features {
        command.arg("--no-default-features");
        for feature in features {
            command.arg(format!("--features={}", feature));
        }
    }

    for name in packages {
        command.arg("-p").arg(name);
    }

    command.arg("--");
    command.args(filters);

    let status = command
        // Disable rust-lang/cargo's cross-compile tests
        .env("CFG_DISABLE_CROSS_TESTS", "1")
        // Relax #![deny(warnings)] in some crates
        .env("RUSTFLAGS", "--cap-lints warn")
        // servo tries to use 'lld-link.exe' on windows, but we don't
        // have lld on our PATH in CI. Override it to use 'link.exe'
        .env("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER", "link.exe")
        .env("CARGO_TARGET_I686_PC_WINDOWS_MSVC_LINKER", "link.exe")
        .current_dir(crate_path)
        .status()
        .unwrap();

    status.success()
}
