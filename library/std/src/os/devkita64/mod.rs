//! devkitA64-specific definitions

#![stable(feature = "raw_ext", since = "1.1.0")]

pub mod fs;
pub mod raw;

extern "C" {
    fn romfsMountSelf(name: *const u8) -> u32;
}

/// This is necessary for getting debug info (needed by backtraces).
pub(crate) fn initialize_romfs() {
    unsafe {
        // This will return an error if there isn't a romfs, which can be safely ignored.
        // SAFETY: We're passing a valid C string to romfsMountSelf.
        let _ = romfsMountSelf(b"romfs\0" as *const _);
    }
}
