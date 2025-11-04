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
    pub(crate) const GENMC_COMMIT: &str = "d9527280bb99f1cef64326b1803ffd952e3880df";

    /// Ensure that a local GenMC repo is present and set to the correct commit.
    /// Return the path of the GenMC repo and whether the checked out commit was changed.
    pub(crate) fn download_genmc() -> (PathBuf, bool) {
        let Ok(genmc_download_path) = PathBuf::from_str(GENMC_DOWNLOAD_PATH);
        let commit_oid = Oid::from_str(GENMC_COMMIT).expect("Commit should be valid.");

        match Repository::open(&genmc_download_path) {
            Ok(repo) => {
                assert_repo_unmodified(&repo);
                if let Ok(head) = repo.head()
                    && let Ok(head_commit) = head.peel_to_commit()
                    && head_commit.id() == commit_oid
                {
                    // Fast path: The expected commit is already checked out.
                    return (genmc_download_path, false);
                }
                // Check if the local repository already contains the commit we need, download it otherwise.
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

        (genmc_download_path, true)
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
            "cargo::warning=GenMC repository remote URL has changed from '{}' to '{GENMC_GITHUB_URL}'",
            remote_url.unwrap_or_default()
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
fn compile_cpp_dependencies(genmc_path: &Path, always_configure: bool) {
    // Give each step a separate build directory to prevent interference.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").as_deref().unwrap());
    let genmc_build_dir = out_dir.join("genmc");
    let interface_build_dir = out_dir.join("miri_genmc");

    // Part 1:
    // Compile the GenMC library using cmake.

    // FIXME(genmc,cargo): Switch to using `CARGO_CFG_DEBUG_ASSERTIONS` once https://github.com/rust-lang/cargo/issues/15760 is completed.
    // Enable/disable additional debug checks, prints and options for GenMC, based on the Rust profile (debug/release)
    let enable_genmc_debug = matches!(std::env::var("PROFILE").as_deref().unwrap(), "debug");

    let mut config = cmake::Config::new(genmc_path);
    config
        .always_configure(always_configure) // We force running the configure step when the GenMC commit changed.
        .out_dir(genmc_build_dir)
        .profile(GENMC_CMAKE_PROFILE)
        .define("GENMC_DEBUG", if enable_genmc_debug { "ON" } else { "OFF" });

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

    // These are all the C++ files we need to compile, which needs to be updated if more C++ files are added to Miri.
    // We use absolute paths since relative paths can confuse IDEs when attempting to go-to-source on a path in a compiler error.
    let cpp_files_base_path = Path::new("cpp/src/");
    let cpp_files = [
        "MiriInterface/EventHandling.cpp",
        "MiriInterface/Exploration.cpp",
        "MiriInterface/Mutex.cpp",
        "MiriInterface/Setup.cpp",
        "MiriInterface/ThreadManagement.cpp",
    ]
    .map(|file| std::path::absolute(cpp_files_base_path.join(file)).unwrap());

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
        .include("./cpp/include")
        .files(&cpp_files)
        .out_dir(interface_build_dir)
        .compile("genmc_interop");

    // Link the Rust-C++ interface library generated by cxx_build:
    println!("cargo::rustc-link-lib=static=genmc_interop");
}

fn main() {
    // Select which path to use for the GenMC repo:
    let (genmc_path, always_configure) = if let Some(genmc_src_path) = option_env!("GENMC_SRC_PATH")
    {
        let genmc_src_path =
            PathBuf::from_str(&genmc_src_path).expect("GENMC_SRC_PATH should contain a valid path");
        assert!(
            genmc_src_path.exists(),
            "GENMC_SRC_PATH={} does not exist!",
            genmc_src_path.display()
        );
        // Rebuild files in the given path change.
        println!("cargo::rerun-if-changed={}", genmc_src_path.display());
        // We disable `always_configure` when working with a local repository,
        // since it increases compile times when working on `genmc-sys`.
        (genmc_src_path, false)
    } else {
        // Download GenMC if required and ensure that the correct commit is checked out.
        // If anything changed in the downloaded repository (e.g., the commit),
        // we set `always_configure` to ensure there are no weird configs from previous builds.
        downloading::download_genmc()
    };

    // Build all required components:
    compile_cpp_dependencies(&genmc_path, always_configure);

    // Only rebuild if anything changes:
    // Note that we don't add the downloaded GenMC repo, since that should never be modified
    // manually. Adding that path here would also trigger an unnecessary rebuild after the repo is
    // cloned (since cargo detects that as a file modification).
    println!("cargo::rerun-if-changed={RUST_CXX_BRIDGE_FILE_PATH}");
    println!("cargo::rerun-if-changed=./src");
    println!("cargo::rerun-if-changed=./cpp");
}
