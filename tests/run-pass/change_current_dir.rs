// ignore-windows: TODO the windows hook is not done yet
// compile-flags: -Zmiri-disable-isolation
use std::env;
use std::path::Path;

fn main() {
    // test that `getcwd` is available
    let cwd = env::current_dir().unwrap();
    let parent = cwd.parent().unwrap_or(&cwd);
    // test that `chdir` is available
    assert!(env::set_current_dir(&Path::new("..")).is_ok());
    // test that `..` goes to the parent directory
    assert_eq!(env::current_dir().unwrap(), parent);
}
