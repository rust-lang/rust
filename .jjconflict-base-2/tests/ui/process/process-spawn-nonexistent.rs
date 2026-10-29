//@ run-pass
//@ needs-subprocess
//@ ignore-fuchsia ErrorKind not translated

use std::io::ErrorKind;
use std::process::Command;

fn main() {
    let result = Command::new("nonexistent").spawn().unwrap_err().kind();

    assert!(matches!(
        result,
        // Under WSL with appendWindowsPath=true, this fails with PermissionDenied
        ErrorKind::NotFound | ErrorKind::PermissionDenied
    ));
}
