//! Tidy check to ensure that the commit SHA of the `src/gcc` submodule is the same as the
//! required GCC version of the GCC codegen backend.

use std::path::Path;
use std::process::Command;

pub fn check(root_path: &Path, compiler_path: &Path, bad: &mut bool) {
    let cg_gcc_version_path = compiler_path.join("rustc_codegen_gcc/libgccjit.version");
    let cg_gcc_version = std::fs::read_to_string(&cg_gcc_version_path)
        .unwrap_or_else(|_| {
            panic!("Cannot read GCC version from {}", cg_gcc_version_path.display())
        })
        .trim()
        .to_string();

    let git_output = Command::new("git")
        .current_dir(root_path)
        .arg("submodule")
        .arg("status")
        // --cached asks for the version that is actually committed in the repository, not the one
        // that is currently checked out.
        .arg("--cached")
        .arg("src/gcc")
        .output()
        .expect("Cannot determine git SHA of the src/gcc checkout");

    // Git is not available or we are in a tarball
    if !git_output.status.success() {
        eprintln!("Cannot figure out the SHA of the GCC submodule");
        return;
    }

    // This can return e.g.
    // -e607be166673a8de9fc07f6f02c60426e556c5f2 src/gcc
    //  e607be166673a8de9fc07f6f02c60426e556c5f2 src/gcc (master-e607be166673a8de9fc07f6f02c60426e556c5f2.e607be)
    // +e607be166673a8de9fc07f6f02c60426e556c5f2 src/gcc (master-e607be166673a8de9fc07f6f02c60426e556c5f2.e607be)
    let git_output = String::from_utf8_lossy(&git_output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string();

    // The SHA can start with + if the submodule is modified or - if it is not checked out.
    let gcc_submodule_sha = git_output.trim_start_matches(['+', '-']);
    if gcc_submodule_sha != cg_gcc_version {
        *bad = true;
        eprintln!(
            r#"Commit SHA of the src/gcc submodule (`{gcc_submodule_sha}`) does not match the required GCC version of the GCC codegen backend (`{cg_gcc_version}`).
Make sure to set the src/gcc submodule to commit {cg_gcc_version}.
The GCC codegen backend commit is configured at {}."#,
            cg_gcc_version_path.display(),
        );
    }
}
