use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Test {
    repo: &'static str,
    name: &'static str,
    sha: &'static str,
    lock: Option<&'static str>,
    packages: &'static [&'static str],
}

const TEST_REPOS: &[Test] = &[
    Test {
        name: "iron",
        repo: "https://github.com/iron/iron",
        sha: "cf056ea5e8052c1feea6141e40ab0306715a2c33",
        lock: None,
        packages: &[],
    },
    Test {
        name: "ripgrep",
        repo: "https://github.com/BurntSushi/ripgrep",
        sha: "3de31f752729525d85a3d1575ac1978733b3f7e7",
        lock: None,
        packages: &[],
    },
    Test {
        name: "tokei",
        repo: "https://github.com/XAMPPRocky/tokei",
        sha: "fdf3f8cb279a7aeac0696c87e5d8b0cd946e4f9e",
        lock: None,
        packages: &[],
    },
    Test {
        name: "xsv",
        repo: "https://github.com/BurntSushi/xsv",
        sha: "3de6c04269a7d315f7e9864b9013451cd9580a08",
        lock: None,
        packages: &[],
    },
    Test {
        name: "servo",
        repo: "https://github.com/servo/servo",
        sha: "caac107ae8145ef2fd20365e2b8fadaf09c2eb3b",
        lock: None,
        // Only test Stylo a.k.a. Quantum CSS, the parts of Servo going into Firefox.
        // This takes much less time to build than all of Servo and supports stable Rust.
        packages: &["selectors"],
    },
];

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let cargo = &args[1];
    let out_dir = Path::new(&args[2]);
    let cargo = &Path::new(cargo);

    for test in TEST_REPOS.iter().rev() {
        test_repo(cargo, out_dir, test);
    }
}

fn test_repo(cargo: &Path, out_dir: &Path, test: &Test) {
    println!("testing {}", test.repo);
    let dir = clone_repo(test, out_dir);
    if let Some(lockfile) = test.lock {
        fs::write(&dir.join("Cargo.lock"), lockfile).unwrap();
    }
    if !run_cargo_test(cargo, &dir, test.packages) {
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
                .arg("master")
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

fn run_cargo_test(cargo_path: &Path, crate_path: &Path, packages: &[&str]) -> bool {
    let mut command = Command::new(cargo_path);
    command.arg("test");
    for name in packages {
        command.arg("-p").arg(name);
    }
    let status = command
        // Disable rust-lang/cargo's cross-compile tests
        .env("CFG_DISABLE_CROSS_TESTS", "1")
        // Relax #![deny(warnings)] in some crates
        .env("RUSTFLAGS", "--cap-lints warn")
        .current_dir(crate_path)
        .status()
        .unwrap();

    status.success()
}
