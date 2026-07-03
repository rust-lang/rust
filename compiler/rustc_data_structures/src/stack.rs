use std::sync::OnceLock;

// This is the amount of bytes that need to be left on the stack before increasing the size.
// It must be at least as large as the stack required by any code that does not call
// `ensure_sufficient_stack`.
const RED_ZONE: usize = 100 * 1024; // 100k

// The initial stack size, stacker will double the size of the stack starting from this value.
// If stacker doesn't support the platform and this value is less than the actual stack size,
// stacker can actually shrink the stack.
// STACK_SIZE is initialized in rustc_interface, based on RUST_MIN_STACK.
pub static STACK_SIZE: OnceLock<usize> = OnceLock::new();
pub const DEFAULT_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Grows the stack on demand to prevent stack overflow. Call this in strategic locations
/// to "break up" recursive calls. E.g. almost any call to `visit_expr` or equivalent can benefit
/// from this.
///
/// Should not be sprinkled around carelessly, as it causes a little bit of overhead.
#[inline]
pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
    let initial_stack_size = STACK_SIZE.get().copied().unwrap_or(DEFAULT_STACK_SIZE);
    stacker::maybe_grow(RED_ZONE, initial_stack_size, f)
}
