//@only-target: linux

// This is a test for calling eventfd2 through a syscall.
// But we do not support this.
fn main() {
    let initval = 0 as libc::c_uint;
    let flags = (libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) as libc::c_int;

    let result = unsafe {
        libc::syscall(libc::SYS_eventfd2, initval, flags) //~ERROR: unsupported operation
    };

    assert_eq!(result, 3); // The first FD provided would be 3.
}
