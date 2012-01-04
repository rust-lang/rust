// GetLastError doesn't seem to work with stack switching
// xfail-test

#[cfg(target_os = "win32")]
native "stdcall" mod kernel32 {
    fn SetLastError(err: uint);
    fn GetLastError() -> uint;
}


#[cfg(target_os = "win32")]
fn main() {
    let expected = 1234u;
    kernel32::SetLastError(expected);
    let actual = kernel32::GetLastError();
    log(error, actual);
    assert (expected == actual);
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
fn main() { }
