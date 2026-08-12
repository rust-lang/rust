#[allow(dead_code)]
pub fn page_size() -> usize {
    // SAFETY: Untriaged.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
