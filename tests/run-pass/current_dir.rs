// ignore-windows: TODO the windows hook is not done yet
// compile-flags: -Zmiri-disable-isolation
use std::env;
use std::path::Path;

fn main() {
    // Test that `getcwd` is available
    let cwd = env::current_dir().unwrap();
    // Test that changing dir to `..` actually sets the current directory to the parent of `cwd`.
    // The only exception here is if `cwd` is the root directory, then changing directory must
    // keep the current directory equal to `cwd`.
    let parent = cwd.parent().unwrap_or(&cwd);
    // Test that `chdir` is available
    assert!(env::set_current_dir(&Path::new("..")).is_ok());
    // Test that `..` goes to the parent directory
    assert_eq!(env::current_dir().unwrap(), parent);
}
