#[cfg(test)]
use stdarch_test::assert_instr;

extern "C" {
    #[link_name = "llvm.wasm.memory.atomic.wait32"]
    fn llvm_atomic_wait_i32(ptr: *mut i32, exp: i32, timeout: i64) -> i32;
    #[link_name = "llvm.wasm.memory.atomic.wait64"]
    fn llvm_atomic_wait_i64(ptr: *mut i64, exp: i64, timeout: i64) -> i32;
    #[link_name = "llvm.wasm.memory.atomic.notify"]
    fn llvm_atomic_notify(ptr: *mut i32, cnt: i32) -> i32;
}

/// Corresponding intrinsic to wasm's [`memory.atomic.wait32` instruction][instr]
///
/// This function, when called, will block the current thread if the memory
/// pointed to by `ptr` is equal to `expression` (performing this action
/// atomically).
///
/// The argument `timeout_ns` is a maxinum number of nanoseconds the calling
/// thread will be blocked for, if it blocks. If the timeout is negative then
/// the calling thread will be blocked forever.
///
/// The calling thread can only be woken up with a call to the `wake` intrinsic
/// once it has been blocked. Changing the memory behind `ptr` will not wake
/// the thread once it's blocked.
///
/// # Return value
///
/// * 0 - indicates that the thread blocked and then was woken up
/// * 1 - the loaded value from `ptr` didn't match `expression`, the thread
///   didn't block
/// * 2 - the thread blocked, but the timeout expired.
///
/// [instr]: https://webassembly.github.io/threads/syntax/instructions.html#syntax-instr-atomic-memory
#[inline]
#[cfg_attr(test, assert_instr(memory.atomic.wait32))]
#[target_feature(enable = "atomics")]
#[doc(alias("memory.atomic.wait32"))]
pub unsafe fn memory_atomic_wait32(ptr: *mut i32, expression: i32, timeout_ns: i64) -> i32 {
    llvm_atomic_wait_i32(ptr, expression, timeout_ns)
}

/// Corresponding intrinsic to wasm's [`memory.atomic.wait64` instruction][instr]
///
/// This function, when called, will block the current thread if the memory
/// pointed to by `ptr` is equal to `expression` (performing this action
/// atomically).
///
/// The argument `timeout_ns` is a maxinum number of nanoseconds the calling
/// thread will be blocked for, if it blocks. If the timeout is negative then
/// the calling thread will be blocked forever.
///
/// The calling thread can only be woken up with a call to the `wake` intrinsic
/// once it has been blocked. Changing the memory behind `ptr` will not wake
/// the thread once it's blocked.
///
/// # Return value
///
/// * 0 - indicates that the thread blocked and then was woken up
/// * 1 - the loaded value from `ptr` didn't match `expression`, the thread
///   didn't block
/// * 2 - the thread blocked, but the timeout expired.
///
/// [instr]: https://webassembly.github.io/threads/syntax/instructions.html#syntax-instr-atomic-memory
#[inline]
#[cfg_attr(test, assert_instr(memory.atomic.wait64))]
#[target_feature(enable = "atomics")]
#[doc(alias("memory.atomic.wait64"))]
pub unsafe fn memory_atomic_wait64(ptr: *mut i64, expression: i64, timeout_ns: i64) -> i32 {
    llvm_atomic_wait_i64(ptr, expression, timeout_ns)
}

/// Corresponding intrinsic to wasm's [`memory.atomic.notify` instruction][instr]
///
/// This function will notify a number of threads blocked on the address
/// indicated by `ptr`. Threads previously blocked with the `i32_atomic_wait`
/// and `i64_atomic_wait` functions above will be woken up.
///
/// The `waiters` argument indicates how many waiters should be woken up (a
/// maximum). If the value is zero no waiters are woken up.
///
/// # Return value
///
/// Returns the number of waiters which were actually notified.
///
/// [instr]: https://webassembly.github.io/threads/syntax/instructions.html#syntax-instr-atomic-memory
#[inline]
#[cfg_attr(test, assert_instr(memory.atomic.notify))]
#[target_feature(enable = "atomics")]
#[doc(alias("memory.atomic.notify"))]
pub unsafe fn memory_atomic_notify(ptr: *mut i32, waiters: u32) -> u32 {
    llvm_atomic_notify(ptr, waiters as i32) as u32
}
