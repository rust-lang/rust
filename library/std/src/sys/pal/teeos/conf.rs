// Hardcoded to return 4096, since `sysconf` is only implemented as a stub.
pub fn page_size() -> usize {
    // unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    4096
}
