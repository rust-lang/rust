use std::path::{Path, PathBuf};
use std::str::FromStr;

// Build script for running Miri with GenMC.
// Check out doc/genmc.md for more info.

/// Path where the downloaded GenMC repository will be stored (relative to the `genmc-sys` directory).
/// Note that this directory is *not* cleaned up automatically by `cargo clean`.
const GENMC_DOWNLOAD_PATH: &str = "./genmc-src/";

/// Name of the library of the GenMC model checker.
const GENMC_MODEL_CHECKER: &str = "genmc_lib";

/// Path where the `cxx_bridge!` macro is used to define the Rust-C++ interface.
const RUST_CXX_BRIDGE_FILE_PATH: &str = "src/lib.rs";

/// The profile with which to build GenMC.
const GENMC_CMAKE_PROFILE: &str = "RelWithDebInfo";

mod downloading {
    use std::path::PathBuf;
    use std::str::FromStr;

    use git2::{Commit, Oid, Remote, Repository, StatusOptions};

    use super::GENMC_DOWNLOAD_PATH;

    /// The GenMC repository the we get our commit from.
    pub(crate) const GENMC_GITHUB_URL: &str = "https://github.com/MPI-SWS/genmc.git";
    /// The GenMC commit we depend on. It must be available on the specified GenMC repository.
    pub(crate) const GENMC_COMMIT: &str = "3438dd2c1202cd4a47ed7881d099abf23e4167ab";

    pub(crate) fn download_genmc() -> PathBuf {
        let Ok(genmc_download_path) = PathBuf::from_str(GENMC_DOWNLOAD_PATH);
        let commit_oid = Oid::from_str(GENMC_COMMIT).expect("Commit should be valid.");

        match Repository::open(&genmc_download_path) {
            Ok(repo) => {
                assert_repo_unmodified(&repo);
                let commit = update_local_repo(&repo, commit_oid);
                checkout_commit(&repo, &commit);
            }
            Err(_) => {
                let repo = clone_remote_repo(&genmc_download_path);
                let Ok(commit) = repo.find_commit(commit_oid) else {
                    panic!(
                        "Cloned GenMC repository does not contain required commit '{GENMC_COMMIT}'"
                    );
                };
                checkout_commit(&repo, &commit);
            }
        };

        genmc_download_path
    }

    fn get_remote(repo: &Repository) -> Remote<'_> {
        let remote = repo.find_remote("origin").unwrap_or_else(|e| {
                panic!(
                    "Could not load commit ({GENMC_COMMIT}) from remote repository '{GENMC_GITHUB_URL}'. Error: {e}"
                );
            });

        // Ensure that the correct remote URL is set.
        let remote_url = remote.url();
        if let Some(remote_url) = remote_url
            && remote_url == GENMC_GITHUB_URL
        {
            return remote;
        }

        // Update remote URL.
        println!(
            "cargo::warning=GenMC repository remote URL has changed from '{remote_url:?}' to '{GENMC_GITHUB_URL}'"
        );
        repo.remote_set_url("origin", GENMC_GITHUB_URL)
            .expect("cannot rename url of remote 'origin'");

        // Reacquire the `Remote`, since `remote_set_url` doesn't update Remote objects already in memory.
        repo.find_remote("origin").unwrap()
    }

    // Check if the required commit exists already, otherwise try fetching it.
    fn update_local_repo(repo: &Repository, commit_oid: Oid) -> Commit<'_> {
        repo.find_commit(commit_oid).unwrap_or_else(|_find_error| {
            println!("GenMC repository at path '{GENMC_DOWNLOAD_PATH}' does not contain commit '{GENMC_COMMIT}'.");
            // The commit is not in the checkout. Try `git fetch` and hope that we find the commit then.
            let mut remote = get_remote(repo);
            remote.fetch(&[GENMC_COMMIT], None, None).expect("Failed to fetch from remote.");

            repo.find_commit(commit_oid)
                .expect("Remote repository should contain expected commit")
        })
    }

    fn clone_remote_repo(genmc_download_path: &PathBuf) -> Repository {
        Repository::clone(GENMC_GITHUB_URL, &genmc_download_path).unwrap_or_else(|e| {
            panic!("Cannot clone GenMC repo from '{GENMC_GITHUB_URL}': {e:?}");
        })
    }

    /// Set the state of the repo to a specific commit
    fn checkout_commit(repo: &Repository, commit: &Commit<'_>) {
        repo.checkout_tree(commit.as_object(), None).expect("Failed to checkout");
        repo.set_head_detached(commit.id()).expect("Failed to set HEAD");
        println!("Successfully set checked out commit {commit:?}");
    }

    /// Check that the downloaded repository is unmodified.
    /// If it is modified, explain that it shouldn't be, and hint at how to do local development with GenMC.
    /// We don't overwrite any changes made to the directory, to prevent data loss.
    fn assert_repo_unmodified(repo: &Repository) {
        let statuses = repo
            .statuses(Some(
                StatusOptions::new()
                    .include_untracked(true)
                    .include_ignored(false)
                    .include_unmodified(false),
            ))
            .expect("should be able to get repository status");
        if statuses.is_empty() {
            return;
        }

        panic!(
            "Downloaded GenMC repository at path '{GENMC_DOWNLOAD_PATH}' has been modified. Please undo any changes made, or delete the '{GENMC_DOWNLOAD_PATH}' directory to have it downloaded again.\n\
            HINT: For local development, set the environment variable 'GENMC_SRC_PATH' to the path of a GenMC repository."
        );
    }
}

// FIXME(genmc,llvm): Remove once the LLVM dependency of the GenMC model checker is removed.
/// The linked LLVM version is in the generated `config.h`` file, which we parse and use to link to LLVM.
/// Returns c++ compiler definitions required for building with/including LLVM, and the include path for LLVM headers.
fn link_to_llvm(config_file: &Path) -> (String, String) {
    /// Search a string for a line matching `//@VARIABLE_NAME: VARIABLE CONTENT`
    fn extract_value<'a>(input: &'a str, name: &str) -> Option<&'a str> {
        input
            .lines()
            .find_map(|line| line.strip_prefix("//@")?.strip_prefix(name)?.strip_prefix(": "))
    }

    let file_content = std::fs::read_to_string(&config_file).unwrap_or_else(|err| {
        panic!("GenMC config file ({}) should exist, but got errror {err:?}", config_file.display())
    });

    let llvm_definitions = extract_value(&file_content, "LLVM_DEFINITIONS")
        .expect("Config file should contain LLVM_DEFINITIONS");
    let llvm_include_dirs = extract_value(&file_content, "LLVM_INCLUDE_DIRS")
        .expect("Config file should contain LLVM_INCLUDE_DIRS");
    let llvm_library_dir = extract_value(&file_content, "LLVM_LIBRARY_DIR")
        .expect("Config file should contain LLVM_LIBRARY_DIR");
    let llvm_config_path = extract_value(&file_content, "LLVM_CONFIG_PATH")
        .expect("Config file should contain LLVM_CONFIG_PATH");

    // Add linker search path.
    let lib_dir = PathBuf::from_str(llvm_library_dir).unwrap();
    println!("cargo::rustc-link-search=native={}", lib_dir.display());

    // Add libraries to link.
    let output = std::process::Command::new(llvm_config_path)
        .arg("--libs") // Print the libraries to link to (space-separated list)
        .output()
        .expect("failed to execute llvm-config");
    let llvm_link_libs =
        String::try_from(output.stdout).expect("llvm-config output should be a valid string");

    for link_lib in llvm_link_libs.trim().split(" ") {
        let link_lib =
            link_lib.strip_prefix("-l").expect("Linker parameter should start with \"-l\"");
        println!("cargo::rustc-link-lib=dylib={link_lib}");
    }

    (llvm_definitions.to_string(), llvm_include_dirs.to_string())
}

/// Build the GenMC model checker library and the Rust-C++ interop library with cxx.rs
fn compile_cpp_dependencies(genmc_path: &Path) {
    // Part 1:
    // Compile the GenMC library using cmake.

    let cmakelists_path = genmc_path.join("CMakeLists.txt");

    // FIXME(genmc,cargo): Switch to using `CARGO_CFG_DEBUG_ASSERTIONS` once https://github.com/rust-lang/cargo/issues/15760 is completed.
    // Enable/disable additional debug checks, prints and options for GenMC, based on the Rust profile (debug/release)
    let enable_genmc_debug = matches!(std::env::var("PROFILE").as_deref().unwrap(), "debug");

    let mut config = cmake::Config::new(cmakelists_path);
    config.profile(GENMC_CMAKE_PROFILE);
    config.define("GENMC_DEBUG", if enable_genmc_debug { "ON" } else { "OFF" });

    // The actual compilation happens here:
    let genmc_install_dir = config.build();

    // Add the model checker library to be linked and tell rustc where to find it:
    let cmake_lib_dir = genmc_install_dir.join("lib").join("genmc");
    println!("cargo::rustc-link-search=native={}", cmake_lib_dir.display());
    println!("cargo::rustc-link-lib=static={GENMC_MODEL_CHECKER}");

    // FIXME(genmc,llvm): Remove once the LLVM dependency of the GenMC model checker is removed.
    let config_file = genmc_install_dir.join("include").join("genmc").join("config.h");
    let (llvm_definitions, llvm_include_dirs) = link_to_llvm(&config_file);

    // Part 2:
    // Compile the cxx_bridge (the link between the Rust and C++ code).

    let genmc_include_dir = genmc_install_dir.join("include").join("genmc");

    // FIXME(genmc,llvm): remove once LLVM dependency is removed.
    // These definitions are parsed into a cmake list and then printed to the config.h file, so they are ';' separated.
    let definitions = llvm_definitions.split(";");

    let mut bridge = cxx_build::bridge("src/lib.rs");
    // FIXME(genmc,cmake): Remove once the GenMC debug setting is available in the config.h file.
    if enable_genmc_debug {
        bridge.define("ENABLE_GENMC_DEBUG", None);
    }
    for definition in definitions {
        bridge.flag(definition);
    }
    bridge
        .opt_level(2)
        .debug(true) // Same settings that GenMC uses (default for cmake `RelWithDebInfo`)
        .warnings(false) // NOTE: enabling this produces a lot of warnings.
        .std("c++23")
        .include(genmc_include_dir)
        .include(llvm_include_dirs)
        .include("./src_cpp")
        .file("./src_cpp/MiriInterface.hpp")
        .file("./src_cpp/MiriInterface.cpp")
        .compile("genmc_interop");

    // Link the Rust-C++ interface library generated by cxx_build:
    println!("cargo::rustc-link-lib=static=genmc_interop");
}

fn main() {
    // Make sure we don't accidentally distribute a binary with GPL code.
    if option_env!("RUSTC_STAGE").is_some() {
        panic!(
            "genmc should not be enabled in the rustc workspace since it includes a GPL dependency"
        );
    }

    // Select which path to use for the GenMC repo:
    let genmc_path = if let Ok(genmc_src_path) = std::env::var("GENMC_SRC_PATH") {
        let genmc_src_path =
            PathBuf::from_str(&genmc_src_path).expect("GENMC_SRC_PATH should contain a valid path");
        assert!(
            genmc_src_path.exists(),
            "GENMC_SRC_PATH={} does not exist!",
            genmc_src_path.display()
        );
        genmc_src_path
    } else {
        downloading::download_genmc()
    };

    // Build all required components:
    compile_cpp_dependencies(&genmc_path);

    // Only rebuild if anything changes:
    // Note that we don't add the downloaded GenMC repo, since that should never be modified
    // manually. Adding that path here would also trigger an unnecessary rebuild after the repo is
    // cloned (since cargo detects that as a file modification).
    println!("cargo::rerun-if-changed={RUST_CXX_BRIDGE_FILE_PATH}");
    println!("cargo::rerun-if-changed=./src");
    println!("cargo::rerun-if-changed=./src_cpp");
}
