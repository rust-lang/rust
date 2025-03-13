//! ABI definitions for symbols exported by <_>-zkvm targets.
//! 
//! Ideally, these should be the minimum viable symbols to support the std-lib.

#![allow(dead_code)]
pub const DIGEST_WORDS: usize = 8;

/// Standard IO file descriptors for use with sys_read and sys_write.
pub mod fileno {
    pub const STDIN: u32 = 0;
    pub const STDOUT: u32 = 1;
    pub const STDERR: u32 = 2;
}

unsafe extern "C" {
    pub fn sys_halt();
    pub fn sys_rand(recv_buf: *mut u8, words: usize);
    pub fn sys_panic(msg_ptr: *const u8, len: usize) -> !;
    pub fn sys_write(fd: u32, write_buf: *const u8, nbytes: usize);
    pub fn sys_getenv(
        recv_buf: *mut u32,
        words: usize,
        varname: *const u8,
        varname_len: usize,
    ) -> usize;
    pub fn sys_argv(out_words: *mut u32, out_nwords: usize, arg_index: usize) -> usize;
    pub fn sys_alloc_aligned(nwords: usize, align: usize) -> *mut u8;
}

// Note: sys_read is not implemented for succinct-zkvm.
// However, it is a function used in the standard library, so we need to
// implement it.
pub unsafe extern "C" fn sys_read(_: u32, _: *mut u8, _: usize) -> usize {
    panic!("sys_read not implemented for succinct");
}
