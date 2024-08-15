//! Random data generation through `getentropy`.
//!
//! Since issue 8 (2024), the POSIX specification mandates the existence of the
//! `getentropy` function, which fills a slice of up to `GETENTROPY_MAX` bytes
//! (256 on all known platforms) with random data. Luckily, this function has
//! already been available on quite some BSDs before that, having appeared with
//! OpenBSD 5.7 and spread from there:
//!
//! platform   | version | man-page
//! -----------|---------|----------
//! OpenBSD    | 5.6     | <https://man.openbsd.org/getentropy.2>
//! FreeBSD    | 12.0    | <https://man.freebsd.org/cgi/man.cgi?query=getentropy&manpath=FreeBSD+15.0-CURRENT>
//! macOS      | 10.12   | <https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/bsd/man/man2/getentropy.2#L4>
//! NetBSD     | 10.0    | <https://man.netbsd.org/getentropy.3>
//! DragonFly  | 6.1     | <https://man.dragonflybsd.org/?command=getentropy&section=3>
//! Illumos    | ?       | <https://www.illumos.org/man/3C/getentropy>
//! Solaris    | ?       | <https://docs.oracle.com/cd/E88353_01/html/E37841/getentropy-2.html>
//!
//! As it is standardized we use it whereever possible, even when `getrandom` is
//! also available. NetBSD even warns that "Applications should avoid getrandom
//! and use getentropy(2) instead; getrandom may be removed from a later
//! release."[^1].
//!
//! [^1]: <https://man.netbsd.org/getrandom.2>

pub fn fill_bytes(bytes: &mut [u8]) {
    // GETENTROPY_MAX isn't defined yet on most platforms, but it's mandated
    // to be at least 256, so just use that as limit.
    for chunk in bytes.chunks_mut(256) {
        let r = unsafe { libc::getentropy(chunk.as_mut_ptr().cast(), chunk.len()) };
        assert_ne!(r, -1, "failed to generate random data");
    }
}
