// ignore-windows
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]

extern crate libc;

fn main() {
    dup_stdout_stderr_test();
}

fn dup_stdout_stderr_test() {
    let bytes = b"hello dup fd\n";
    unsafe {
        let new_stdout = libc::fcntl(1, libc::F_DUPFD, 0);
        let new_stderr = libc::fcntl(2, libc::F_DUPFD, 0);
        libc::write(new_stdout, bytes.as_ptr() as *const libc::c_void, bytes.len());
        libc::write(new_stderr, bytes.as_ptr() as *const libc::c_void, bytes.len());
    }
}
