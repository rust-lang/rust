// This is the amount of bytes that need to be left on the stack before increasing the size.
// It must be at least as large as the stack required by any code that does not call
// `ensure_sufficient_stack`.
const RED_ZONE: usize = 100 * 1024; // 100k

// Only the first stack that is pushed, grows exponentially (2^n * STACK_PER_RECURSION) from then
// on. This flag has performance relevant characteristics. Don't set it too high.
#[cfg(not(target_os = "aix"))]
const STACK_PER_RECURSION: usize = 1024 * 1024; // 1MB
// LLVM for AIX doesn't feature TCO, increase recursion size for workaround.
#[cfg(target_os = "aix")]
const STACK_PER_RECURSION: usize = 16 * 1024 * 1024; // 16MB

/// Grows the stack on demand to prevent stack overflow. Call this in strategic locations
/// to "break up" recursive calls. E.g. almost any call to `visit_expr` or equivalent can benefit
/// from this.
///
/// Should not be sprinkled around carelessly, as it causes a little bit of overhead.
#[inline]
pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(RED_ZONE, STACK_PER_RECURSION, f)
}
