//! Random data generation with `arc4random_buf`.
//!
//! Contrary to its name, `arc4random` doesn't actually use the horribly-broken
//! RC4 cypher anymore, at least not on modern systems, but rather something
//! like ChaCha20 with continual reseeding from the OS. That makes it an ideal
//! source of large quantities of cryptographically secure data, which is exactly
//! what we need for `DefaultRandomSource`. Unfortunately, it's not available
//! on all UNIX systems, most notably Linux (until recently, but it's just a
//! wrapper for `getrandom`. Since we need to hook into `getrandom` directly
//! for `HashMap` keys anyway, we just keep our version).

#[cfg(not(any(
    target_os = "haiku",
    target_os = "illumos",
    target_os = "solaris",
    target_os = "vita",
)))]
use libc::arc4random_buf;

// FIXME: move these to libc (except Haiku, that one needs to link to libbsd.so).
#[cfg(any(
    target_os = "haiku", // See https://git.haiku-os.org/haiku/tree/headers/compatibility/bsd/stdlib.h
    target_os = "illumos", // See https://www.illumos.org/man/3C/arc4random
    target_os = "solaris", // See https://docs.oracle.com/cd/E88353_01/html/E37843/arc4random-3c.html
    target_os = "vita", // See https://github.com/vitasdk/newlib/blob/b89e5bc183b516945f9ee07eef483ecb916e45ff/newlib/libc/include/stdlib.h#L74
))]
#[cfg_attr(target_os = "haiku", link(name = "bsd"))]
extern "C" {
    fn arc4random_buf(buf: *mut core::ffi::c_void, nbytes: libc::size_t);
}

pub fn fill_bytes(bytes: &mut [u8]) {
    unsafe { arc4random_buf(bytes.as_mut_ptr().cast(), bytes.len()) }
}
