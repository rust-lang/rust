//! Random data generation through `getentropy`.
//!
//! Since issue 8 (2024), the POSIX specification mandates the existence of the
//! `getentropy` function, which fills a slice of up to `GETENTROPY_MAX` bytes
//! (256 on all known platforms) with random data. Unfortunately, it's only
//! meant to be used to seed other CPRNGs, which we don't have, so we only use
//! it where `arc4random_buf` and friends aren't available or secure (currently
//! that's only the case on Emscripten).

pub fn fill_bytes(bytes: &mut [u8]) {
    // GETENTROPY_MAX isn't defined yet on most platforms, but it's mandated
    // to be at least 256, so just use that as limit.
    for chunk in bytes.chunks_mut(256) {
        let r = unsafe { libc::getentropy(chunk.as_mut_ptr().cast(), chunk.len()) };
        assert_ne!(r, -1, "failed to generate random data");
    }
}
