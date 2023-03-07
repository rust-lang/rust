pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    let p = unsafe {
        libc::memchr(
            haystack.as_ptr() as *const libc::c_void,
            needle as libc::c_int,
            haystack.len(),
        )
    };
    if p.is_null() { None } else { Some(p as usize - (haystack.as_ptr() as usize)) }
}

pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    let p = unsafe {
        libc::memrchr(
            haystack.as_ptr() as *const libc::c_void,
            needle as libc::c_int,
            haystack.len(),
        )
    };
    if p.is_null() { None } else { Some(p as usize - (haystack.as_ptr() as usize)) }
}
