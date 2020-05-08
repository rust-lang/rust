//! This crate contains a single public function
//! [`get_path_for_executable`](fn.get_path_for_executable.html).
//! See docs there for more information.
use std::{env, iter, path::PathBuf};

pub fn cargo() -> PathBuf {
    get_path_for_executable("cargo")
}

pub fn rustc() -> PathBuf {
    get_path_for_executable("rustc")
}

pub fn rustup() -> PathBuf {
    get_path_for_executable("rustup")
}

/// Return a `PathBuf` to use for the given executable.
///
/// E.g., `get_path_for_executable("cargo")` may return just `cargo` if that
/// gives a valid Cargo executable; or it may return a full path to a valid
/// Cargo.
fn get_path_for_executable(executable_name: &'static str) -> PathBuf {
    // The current implementation checks three places for an executable to use:
    // 1) Appropriate environment variable (erroring if this is set but not a usable executable)
    //      example: for cargo, this checks $CARGO environment variable; for rustc, $RUSTC; etc
    // 2) `<executable_name>`
    //      example: for cargo, this tries just `cargo`, which will succeed if `cargo` is on the $PATH
    // 3) `~/.cargo/bin/<executable_name>`
    //      example: for cargo, this tries ~/.cargo/bin/cargo
    //      It seems that this is a reasonable place to try for cargo, rustc, and rustup
    let env_var = executable_name.to_ascii_uppercase();
    if let Some(path) = env::var_os(&env_var) {
        return path.into();
    }

    if lookup_in_path(executable_name) {
        return executable_name.into();
    }

    if let Some(mut path) = home::home_dir() {
        path.push(".cargo");
        path.push("bin");
        path.push(executable_name);
        if path.is_file() {
            return path;
        }
    }
    executable_name.into()
}

fn lookup_in_path(exec: &str) -> bool {
    let paths = env::var_os("PATH").unwrap_or_default();
    let mut candidates = env::split_paths(&paths).flat_map(|path| {
        let candidate = path.join(&exec);
        let with_exe = if env::consts::EXE_EXTENSION == "" {
            None
        } else {
            Some(candidate.with_extension(env::consts::EXE_EXTENSION))
        };
        iter::once(candidate).chain(with_exe)
    });
    candidates.any(|it| it.is_file())
}
