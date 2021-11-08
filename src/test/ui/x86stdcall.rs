// run-pass
// ignore-wasm32-bare no libc to test ffi with
// ignore-sgx no libc
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

#[cfg(any(target_os = "android",
          target_os = "dragonfly",
          target_os = "emscripten",
          target_os = "freebsd",
          target_os = "fuchsia",
          target_os = "illumos",
          target_os = "linux",
          target_os = "macos",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "solaris",
          target_os = "vxworks"))]
pub fn main() { }
