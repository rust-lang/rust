//! Intrinsics associated with WebAssembly's upcoming threads proposal.
//!
//! These intrinsics are all unstable because they're not actually stable in
//! WebAssembly itself yet. The signatures may change as [the
//! specification][spec] is updated.
//!
//! [spec]: https://github.com/WebAssembly/threads

#![cfg(any(target_feature = "atomics", dox))]

#[cfg(test)]
use stdsimd_test::assert_instr;
#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test;

extern "C" {
    #[link_name = "llvm.wasm.atomic.wait.i32"]
    fn llvm_atomic_wait_i32(ptr: *mut i32, exp: i32, timeout: i64) -> i32;
    #[link_name = "llvm.wasm.atomic.wait.i64"]
    fn llvm_atomic_wait_i64(ptr: *mut i64, exp: i64, timeout: i64) -> i32;
    #[link_name = "llvm.wasm.atomic.notify"]
    fn llvm_atomic_notify(ptr: *mut i32, cnt: i32) -> i32;
}

/// Corresponding intrinsic to wasm's [`i32.atomic.wait` instruction][instr]
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
/// once it has been blocked. Changing the memory behind `ptr` will not wake the
/// thread once it's blocked.
///
/// # Return value
///
/// * 0 - indicates that the thread blocked and then was woken up
/// * 1 - the loaded value from `ptr` didn't match `expression`, the thread
///   didn't block
/// * 2 - the thread blocked, but the timeout expired.
///
/// # Availability
///
/// This intrinsic is only available **when the standard library itself is
/// compiled with the `atomics` target feature**. This version of the standard
/// library is not obtainable via `rustup`, but rather will require the standard
/// library to be compiled from source.
///
/// [instr]: https://github.com/WebAssembly/threads/blob/master/proposals/threads/Overview.md#wait
#[inline]
#[cfg_attr(test, assert_instr("i32.atomic.wait"))]
pub unsafe fn wait_i32(ptr: *mut i32, expression: i32, timeout_ns: i64) -> i32 {
    llvm_atomic_wait_i32(ptr, expression, timeout_ns)
}

/// Corresponding intrinsic to wasm's [`i64.atomic.wait` instruction][instr]
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
/// once it has been blocked. Changing the memory behind `ptr` will not wake the
/// thread once it's blocked.
///
/// # Return value
///
/// * 0 - indicates that the thread blocked and then was woken up
/// * 1 - the loaded value from `ptr` didn't match `expression`, the thread
///   didn't block
/// * 2 - the thread blocked, but the timeout expired.
///
/// # Availability
///
/// This intrinsic is only available **when the standard library itself is
/// compiled with the `atomics` target feature**. This version of the standard
/// library is not obtainable via `rustup`, but rather will require the standard
/// library to be compiled from source.
///
/// [instr]: https://github.com/WebAssembly/threads/blob/master/proposals/threads/Overview.md#wait
#[inline]
#[cfg_attr(test, assert_instr("i64.atomic.wait"))]
pub unsafe fn wait_i64(ptr: *mut i64, expression: i64, timeout_ns: i64) -> i32 {
    llvm_atomic_wait_i64(ptr, expression, timeout_ns)
}

/// Corresponding intrinsic to wasm's [`atomic.wake` instruction][instr]
///
/// This function will wake up a number of threads blocked on the address
/// indicated by `ptr`. Threads previously blocked with the `wait_i32` and
/// `wait_i64` functions above will be woken up.
///
/// The `waiters` argument indicates how many waiters should be woken up (a
/// maximum). If the value is negative all waiters are woken up, and if the
/// value is zero no waiters are woken up.
///
/// # Return value
///
/// Returns the number of waiters which were actually woken up.
///
/// # Availability
///
/// This intrinsic is only available **when the standard library itself is
/// compiled with the `atomics` target feature**. This version of the standard
/// library is not obtainable via `rustup`, but rather will require the standard
/// library to be compiled from source.
///
/// [instr]: https://github.com/WebAssembly/threads/blob/master/proposals/threads/Overview.md#wake
#[inline]
#[cfg_attr(test, assert_instr("atomic.wake"))]
pub unsafe fn wake(ptr: *mut i32, waiters: i32) -> i32 {
    llvm_atomic_notify(ptr, waiters)
}
