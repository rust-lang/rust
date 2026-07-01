//! ABI definitions for symbols exported by OpenVM.

// We provide the ABI so that the OpenVM-specific implementations can be provided
// by linking the openvm crate without introducing the crate as a dependency here
#![allow(dead_code)]

/// Standard IO file descriptors for use with sys_read and sys_write.
pub mod fileno {
    pub const STDIN: u32 = 0;
    pub const STDOUT: u32 = 1;
    pub const STDERR: u32 = 2;
    pub const JOURNAL: u32 = 3;
}

unsafe extern "C" {
    // Wrappers around syscalls provided by OpenVM:
    pub fn sys_halt();
    pub fn sys_rand(recv_buf: *mut u32, words: usize);
    pub fn sys_panic(msg_ptr: *const u8, len: usize) -> !;
    pub fn sys_log(msg_ptr: *const u8, len: usize);
    pub fn sys_read(fd: u32, recv_buf: *mut u8, nrequested: usize) -> usize;
    pub fn sys_write(fd: u32, write_buf: *const u8, nbytes: usize);
    pub fn sys_getenv(
        recv_buf: *mut u32,
        words: usize,
        varname: *const u8,
        varname_len: usize,
    ) -> usize;
    pub fn sys_argc() -> usize;
    pub fn sys_argv(out_words: *mut u32, out_nwords: usize, arg_index: usize) -> usize;

    // Allocate memory from global HEAP.
    pub fn sys_alloc_words(nwords: usize) -> *mut u32;
    pub fn sys_alloc_aligned(nwords: usize, align: usize) -> *mut u8;
}
