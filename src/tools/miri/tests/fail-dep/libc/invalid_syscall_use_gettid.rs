//@only-target: linux android

fn main() {
    // gettid takes no parameters, this should be rejected
    unsafe { libc::syscall(libc::SYS_gettid, 0) }; //~ERROR: incorrect number of varidic arguments
}
