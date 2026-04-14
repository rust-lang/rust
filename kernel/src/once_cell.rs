//! Minimal no_std OnceCell primitive.
//!
//! This provides single-assignment semantics with explicit panics on:
//! - Double initialization (footgun prevention for SMP/refactor)
//! - Access before initialization (clear error message)

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// A cell that can be written to exactly once.
///
/// Unlike `spin::Once`, this panics on double-init rather than silently
/// ignoring subsequent calls.
pub struct OnceCell<T> {
    initialized: AtomicBool,
    value: UnsafeCell<Option<T>>,
}

// SAFETY: OnceCell uses atomic operations for synchronization.
// The interior value is only written once during init (single-writer
// on a single core), then read-only thereafter. This is safe for the
// kernel's single-core initialization sequence.
unsafe impl<T> Send for OnceCell<T> {}
unsafe impl<T> Sync for OnceCell<T> {}

impl<T> OnceCell<T> {
    /// Creates a new uninitialized `OnceCell`.
    pub const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            value: UnsafeCell::new(None),
        }
    }

    /// Sets the value exactly once.
    ///
    /// # Panics
    /// Panics if the cell has already been initialized.
    pub fn set(&self, value: T) {
        let was_init = self.initialized.swap(true, Ordering::AcqRel);
        if was_init {
            panic!("OnceCell::set() called on already-initialized cell");
        }
        // SAFETY: We just atomically claimed initialization rights
        unsafe {
            *self.value.get() = Some(value);
        }
    }

    /// Gets a reference to the value.
    ///
    /// # Panics
    /// Panics if the cell has not been initialized.
    pub fn get(&self) -> &T {
        if !self.initialized.load(Ordering::Acquire) {
            panic!("OnceCell::get() called before initialization");
        }
        // SAFETY: initialized flag is true, value was written
        unsafe {
            (*self.value.get())
                .as_ref()
                .expect("OnceCell invariant violated: initialized but no value")
        }
    }

    /// Returns `true` if the cell has been initialized.
    #[allow(dead_code)]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}
