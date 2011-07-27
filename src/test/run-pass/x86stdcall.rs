// xfail-stage0

#[cfg(target_os = "win32")]
native "x86stdcall" mod kernel32 {
    fn SetLastError(err: uint);
    fn GetLastError() -> uint;
}

#[cfg(target_os = "win32")]
fn main() {
    let expected = 10u;
    kernel32::SetLastError(expected);
    let actual = kernel32::GetLastError();
    assert (expected == actual);
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
fn main() { }