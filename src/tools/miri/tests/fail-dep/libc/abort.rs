fn main() {
    unsafe { libc::abort() };
    //~^ error: abnormal termination: the program aborted execution
}
