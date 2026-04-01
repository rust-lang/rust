//@compile-flags: -Zmiri-disable-isolation
use std::env;
use std::io::ErrorKind;

fn main() {
    // Test that `getcwd` is available and an absolute path
    let cwd = env::current_dir().unwrap();
    assert!(cwd.is_absolute(), "cwd {:?} is not absolute", cwd);
    // Test that changing dir to `..` actually sets the current directory to the parent of `cwd`.
    // The only exception here is if `cwd` is the root directory, then changing directory must
    // keep the current directory equal to `cwd`.
    let parent = cwd.parent().unwrap_or(&cwd);
    // Test that `chdir` is available
    assert!(env::set_current_dir("..").is_ok());
    // Test that `..` goes to the parent directory
    assert_eq!(env::current_dir().unwrap(), parent);
    // Test that `chdir` to a non-existing directory returns a proper error
    assert_eq!(env::set_current_dir("thisdoesnotexist").unwrap_err().kind(), ErrorKind::NotFound);
}
