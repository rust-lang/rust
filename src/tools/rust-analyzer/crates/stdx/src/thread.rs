//! A utility module for working with threads that automatically joins threads upon drop
//! and abstracts over operating system quality of service (QoS) APIs
//! through the concept of a “thread intent”.
//!
//! The intent of a thread is frozen at thread creation time,
//! i.e. there is no API to change the intent of a thread once it has been spawned.
//!
//! As a system, rust-analyzer should have the property that
//! old manual scheduling APIs are replaced entirely by QoS.
//! To maintain this invariant, we panic when it is clear that
//! old scheduling APIs have been used.
//!
//! Moreover, we also want to ensure that every thread has an intent set explicitly
//! to force a decision about its importance to the system.
//! Thus, [`ThreadIntent`] has no default value
//! and every entry point to creating a thread requires a [`ThreadIntent`] upfront.

use std::fmt;

mod intent;
mod pool;

pub use intent::ThreadIntent;
pub use pool::Pool;

pub fn spawn<F, T>(intent: ThreadIntent, f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    Builder::new(intent).spawn(f).expect("failed to spawn thread")
}

pub struct Builder {
    intent: ThreadIntent,
    inner: jod_thread::Builder,
    allow_leak: bool,
}

impl Builder {
    pub fn new(intent: ThreadIntent) -> Builder {
        Builder { intent, inner: jod_thread::Builder::new(), allow_leak: false }
    }

    pub fn name(self, name: String) -> Builder {
        Builder { inner: self.inner.name(name), ..self }
    }

    pub fn stack_size(self, size: usize) -> Builder {
        Builder { inner: self.inner.stack_size(size), ..self }
    }

    pub fn allow_leak(self, b: bool) -> Builder {
        Builder { allow_leak: b, ..self }
    }

    pub fn spawn<F, T>(self, f: F) -> std::io::Result<JoinHandle<T>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        let inner_handle = self.inner.spawn(move || {
            self.intent.apply_to_current_thread();
            f()
        })?;

        Ok(JoinHandle { inner: Some(inner_handle), allow_leak: self.allow_leak })
    }
}

pub struct JoinHandle<T = ()> {
    // `inner` is an `Option` so that we can
    // take ownership of the contained `JoinHandle`.
    inner: Option<jod_thread::JoinHandle<T>>,
    allow_leak: bool,
}

impl<T> JoinHandle<T> {
    pub fn join(mut self) -> T {
        self.inner.take().unwrap().join()
    }
}

impl<T> Drop for JoinHandle<T> {
    fn drop(&mut self) {
        if !self.allow_leak {
            return;
        }

        if let Some(join_handle) = self.inner.take() {
            join_handle.detach();
        }
    }
}

impl<T> fmt::Debug for JoinHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("JoinHandle { .. }")
    }
}
