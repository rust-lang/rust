//! ABI definitions for symbols exported by risc0-zkvm-platform.

// Included here so we don't have to depend on risc0-zkvm-platform.
//
// FIXME: Should we move this to the "libc" crate?  It seems like other
// architectures put a lot of this kind of stuff there.  But there's
// currently no risc0 fork of the libc crate, so we'd either have to
// fork it or upstream it.

#![allow(dead_code)]
pub const DIGEST_WORDS: usize = 8;

/// Standard IO file descriptors for use with sys_read and sys_write.
pub mod fileno {
    pub const STDIN: u32 = 0;
    pub const STDOUT: u32 = 1;
    pub const STDERR: u32 = 2;
    pub const JOURNAL: u32 = 3;
}

extern "C" {
    // Wrappers around syscalls provided by risc0-zkvm-platform:
    pub fn sys_halt();
    pub fn sys_output(output_id: u32, output_value: u32);
    pub fn sys_sha_compress(
        out_state: *mut [u32; DIGEST_WORDS],
        in_state: *const [u32; DIGEST_WORDS],
        block1_ptr: *const [u32; DIGEST_WORDS],
        block2_ptr: *const [u32; DIGEST_WORDS],
    );
    pub fn sys_sha_buffer(
        out_state: *mut [u32; DIGEST_WORDS],
        in_state: *const [u32; DIGEST_WORDS],
        buf: *const u8,
        count: u32,
    );
    pub fn sys_rand(recv_buf: *mut u32, words: usize);
    pub fn sys_panic(msg_ptr: *const u8, len: usize) -> !;
    pub fn sys_log(msg_ptr: *const u8, len: usize);
    pub fn sys_cycle_count() -> usize;
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
