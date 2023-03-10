//! The following module declarations are outside cfg_if because the internal
//! `__thread_local_internal` macro does not seem to be exported properly when using cfg_if
#![unstable(feature = "thread_local_internals", reason = "should not be necessary", issue = "none")]

#[cfg(all(target_thread_local, not(all(target_family = "wasm", not(target_feature = "atomics")))))]
mod fast_local;
#[cfg(all(
    not(target_thread_local),
    not(all(target_family = "wasm", not(target_feature = "atomics")))
))]
mod os_local;
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
mod static_local;

#[cfg(not(test))]
cfg_if::cfg_if! {
    if #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))] {
        #[doc(hidden)]
        pub use static_local::statik::Key;
    } else if #[cfg(all(target_thread_local, not(all(target_family = "wasm", not(target_feature = "atomics")))))] {
        #[doc(hidden)]
        pub use fast_local::fast::Key;
    } else if #[cfg(all(not(target_thread_local), not(all(target_family = "wasm", not(target_feature = "atomics")))))] {
        #[doc(hidden)]
        pub use os_local::os::Key;
    }
}

#[doc(hidden)]
#[cfg(test)]
pub use realstd::thread::__LocalKeyInner as Key;

mod lazy {
    use crate::cell::UnsafeCell;
    use crate::hint;
    use crate::mem;

    pub struct LazyKeyInner<T> {
        inner: UnsafeCell<Option<T>>,
    }

    impl<T> LazyKeyInner<T> {
        pub const fn new() -> LazyKeyInner<T> {
            LazyKeyInner { inner: UnsafeCell::new(None) }
        }

        pub unsafe fn get(&self) -> Option<&'static T> {
            // SAFETY: The caller must ensure no reference is ever handed out to
            // the inner cell nor mutable reference to the Option<T> inside said
            // cell. This make it safe to hand a reference, though the lifetime
            // of 'static is itself unsafe, making the get method unsafe.
            unsafe { (*self.inner.get()).as_ref() }
        }

        /// The caller must ensure that no reference is active: this method
        /// needs unique access.
        pub unsafe fn initialize<F: FnOnce() -> T>(&self, init: F) -> &'static T {
            // Execute the initialization up front, *then* move it into our slot,
            // just in case initialization fails.
            let value = init();
            let ptr = self.inner.get();

            // SAFETY:
            //
            // note that this can in theory just be `*ptr = Some(value)`, but due to
            // the compiler will currently codegen that pattern with something like:
            //
            //      ptr::drop_in_place(ptr)
            //      ptr::write(ptr, Some(value))
            //
            // Due to this pattern it's possible for the destructor of the value in
            // `ptr` (e.g., if this is being recursively initialized) to re-access
            // TLS, in which case there will be a `&` and `&mut` pointer to the same
            // value (an aliasing violation). To avoid setting the "I'm running a
            // destructor" flag we just use `mem::replace` which should sequence the
            // operations a little differently and make this safe to call.
            //
            // The precondition also ensures that we are the only one accessing
            // `self` at the moment so replacing is fine.
            unsafe {
                let _ = mem::replace(&mut *ptr, Some(value));
            }

            // SAFETY: With the call to `mem::replace` it is guaranteed there is
            // a `Some` behind `ptr`, not a `None` so `unreachable_unchecked`
            // will never be reached.
            unsafe {
                // After storing `Some` we want to get a reference to the contents of
                // what we just stored. While we could use `unwrap` here and it should
                // always work it empirically doesn't seem to always get optimized away,
                // which means that using something like `try_with` can pull in
                // panicking code and cause a large size bloat.
                match *ptr {
                    Some(ref x) => x,
                    None => hint::unreachable_unchecked(),
                }
            }
        }

        /// The other methods hand out references while taking &self.
        /// As such, callers of this method must ensure no `&` and `&mut` are
        /// available and used at the same time.
        #[allow(unused)]
        pub unsafe fn take(&mut self) -> Option<T> {
            // SAFETY: See doc comment for this method.
            unsafe { (*self.inner.get()).take() }
        }
    }
}
