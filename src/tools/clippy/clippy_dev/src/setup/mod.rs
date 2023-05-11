pub mod git_hook;
pub mod intellij;
pub mod vscode;

use std::path::Path;

const CLIPPY_DEV_DIR: &str = "clippy_dev";

/// This function verifies that the tool is being executed in the clippy directory.
/// This is useful to ensure that setups only modify Clippy's resources. The verification
/// is done by checking that `clippy_dev` is a sub directory of the current directory.
///
/// It will print an error message and return `false` if the directory could not be
/// verified.
fn verify_inside_clippy_dir() -> bool {
    let path = Path::new(CLIPPY_DEV_DIR);
    if path.exists() && path.is_dir() {
        true
    } else {
        eprintln!("error: unable to verify that the working directory is clippy's directory");
        false
    }
}
