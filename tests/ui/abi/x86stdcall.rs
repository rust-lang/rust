//@ run-pass
//@ only-windows
//@ ignore-backends: gcc
// GetLastError doesn't seem to work with stack switching

#[cfg(windows)]
mod kernel32 {
    extern "system" {
        pub fn SetLastError(err: usize);
        pub fn GetLastError() -> usize;
    }
}

#[cfg(windows)]
pub fn main() {
    unsafe {
        let expected = 1234;
        kernel32::SetLastError(expected);
        let actual = kernel32::GetLastError();
        println!("actual = {}", actual);
        assert_eq!(expected, actual);
    }
}
