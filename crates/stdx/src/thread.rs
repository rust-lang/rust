//! A utility module for working with threads that automatically joins threads upon drop
//! and provides functionality for interfacing with operating system quality of service (QoS) APIs.
//!
//! As a system, rust-analyzer should have the property that
//! old manual scheduling APIs are replaced entirely by QoS.
//! To maintain this invariant, we panic when it is clear that
//! old scheduling APIs have been used.
//!
//! Moreover, we also want to ensure that every thread has a QoS set explicitly
//! to force a decision about its importance to the system.
//! Thus, [`QoSClass`] has no default value
//! and every entry point to creating a thread requires a [`QoSClass`] upfront.

use std::fmt;

pub fn spawn<F, T>(qos_class: QoSClass, f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    Builder::new(qos_class).spawn(f).expect("failed to spawn thread")
}

pub struct Builder {
    qos_class: QoSClass,
    inner: jod_thread::Builder,
    allow_leak: bool,
}

impl Builder {
    pub fn new(qos_class: QoSClass) -> Builder {
        Builder { qos_class, inner: jod_thread::Builder::new(), allow_leak: false }
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
            set_current_thread_qos_class(self.qos_class);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
// Please maintain order from least to most priority for the derived `Ord` impl.
pub enum QoSClass {
    // Documentation adapted from https://github.com/apple-oss-distributions/libpthread/blob/67e155c94093be9a204b69637d198eceff2c7c46/include/sys/qos.h#L55
    //
    /// TLDR: invisible maintenance tasks
    ///
    /// Contract:
    ///
    /// * **You do not care about how long it takes for work to finish.**
    /// * **You do not care about work being deferred temporarily.**
    ///   (e.g. if the device’s battery is in a critical state)
    ///
    /// Examples:
    ///
    /// * in a video editor:
    ///   creating periodic backups of project files
    /// * in a browser:
    ///   cleaning up cached sites which have not been accessed in a long time
    /// * in a collaborative word processor:
    ///   creating a searchable index of all documents
    ///
    /// Use this QoS class for background tasks
    /// which the user did not initiate themselves
    /// and which are invisible to the user.
    /// It is expected that this work will take significant time to complete:
    /// minutes or even hours.
    ///
    /// This QoS class provides the most energy and thermally-efficient execution possible.
    /// All other work is prioritized over background tasks.
    Background,

    /// TLDR: tasks that don’t block using your app
    ///
    /// Contract:
    ///
    /// * **Your app remains useful even as the task is executing.**
    ///
    /// Examples:
    ///
    /// * in a video editor:
    ///   exporting a video to disk –
    ///   the user can still work on the timeline
    /// * in a browser:
    ///   automatically extracting a downloaded zip file –
    ///   the user can still switch tabs
    /// * in a collaborative word processor:
    ///   downloading images embedded in a document –
    ///   the user can still make edits
    ///
    /// Use this QoS class for tasks which
    /// may or may not be initiated by the user,
    /// but whose result is visible.
    /// It is expected that this work will take a few seconds to a few minutes.
    /// Typically your app will include a progress bar
    /// for tasks using this class.
    ///
    /// This QoS class provides a balance between
    /// performance, responsiveness and efficiency.
    Utility,

    /// TLDR: tasks that block using your app
    ///
    /// Contract:
    ///
    /// * **You need this work to complete
    ///   before the user can keep interacting with your app.**
    /// * **Your work will not take more than a few seconds to complete.**
    ///
    /// Examples:
    ///
    /// * in a video editor:
    ///   opening a saved project
    /// * in a browser:
    ///   loading a list of the user’s bookmarks and top sites
    ///   when a new tab is created
    /// * in a collaborative word processor:
    ///   running a search on the document’s content
    ///
    /// Use this QoS class for tasks which were initiated by the user
    /// and block the usage of your app while they are in progress.
    /// It is expected that this work will take a few seconds or less to complete;
    /// not long enough to cause the user to switch to something else.
    /// Your app will likely indicate progress on these tasks
    /// through the display of placeholder content or modals.
    ///
    /// This QoS class is not energy-efficient.
    /// Rather, it provides responsiveness
    /// by prioritizing work above other tasks on the system
    /// except for critical user-interactive work.
    UserInitiated,

    /// TLDR: render loops and nothing else
    ///
    /// Contract:
    ///
    /// * **You absolutely need this work to complete immediately
    ///   or your app will appear to freeze.**
    /// * **Your work will always complete virtually instantaneously.**
    ///
    /// Examples:
    ///
    /// * the main thread in a GUI application
    /// * the update & render loop in a game
    /// * a secondary thread which progresses an animation
    ///
    /// Use this QoS class for any work which, if delayed,
    /// will make your user interface unresponsive.
    /// It is expected that this work will be virtually instantaneous.
    ///
    /// This QoS class is not energy-efficient.
    /// Specifying this class is a request to run with
    /// nearly all available system CPU and I/O bandwidth even under contention.
    UserInteractive,
}

pub const IS_QOS_AVAILABLE: bool = imp::IS_QOS_AVAILABLE;

pub fn set_current_thread_qos_class(class: QoSClass) {
    imp::set_current_thread_qos_class(class)
}

pub fn get_current_thread_qos_class() -> Option<QoSClass> {
    imp::get_current_thread_qos_class()
}

// All Apple platforms use XNU as their kernel
// and thus have the concept of QoS.
#[cfg(target_vendor = "apple")]
mod imp {
    use super::QoSClass;

    pub(super) const IS_QOS_AVAILABLE: bool = true;

    pub(super) fn set_current_thread_qos_class(class: QoSClass) {
        let c = match class {
            QoSClass::UserInteractive => libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
            QoSClass::UserInitiated => libc::qos_class_t::QOS_CLASS_USER_INITIATED,
            QoSClass::Utility => libc::qos_class_t::QOS_CLASS_UTILITY,
            QoSClass::Background => libc::qos_class_t::QOS_CLASS_BACKGROUND,
        };

        let code = unsafe { libc::pthread_set_qos_class_self_np(c, 0) };

        if code == 0 {
            return;
        }

        let errno = unsafe { *libc::__error() };

        match errno {
            libc::EPERM => {
                // This thread has been excluded from the QoS system
                // due to a previous call to a function such as `pthread_setschedparam`
                // which is incompatible with QoS.
                //
                // Panic instead of returning an error
                // to maintain the invariant that we only use QoS APIs.
                panic!("tried to set QoS of thread which has opted out of QoS (os error {errno})")
            }

            libc::EINVAL => {
                // This is returned if we pass something other than a qos_class_t
                // to `pthread_set_qos_class_self_np`.
                //
                // This is impossible, so again panic.
                unreachable!(
                    "invalid qos_class_t value was passed to pthread_set_qos_class_self_np"
                )
            }

            _ => {
                // `pthread_set_qos_class_self_np`’s documentation
                // does not mention any other errors.
                unreachable!("`pthread_set_qos_class_self_np` returned unexpected error {errno}")
            }
        }
    }

    pub(super) fn get_current_thread_qos_class() -> Option<QoSClass> {
        let current_thread = unsafe { libc::pthread_self() };
        let mut qos_class_raw = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
        let code = unsafe {
            libc::pthread_get_qos_class_np(current_thread, &mut qos_class_raw, std::ptr::null_mut())
        };

        if code != 0 {
            // `pthread_get_qos_class_np`’s documentation states that
            // an error value is placed into errno if the return code is not zero.
            // However, it never states what errors are possible.
            // Inspecting the source[0] shows that, as of this writing, it always returns zero.
            //
            // Whatever errors the function could report in future are likely to be
            // ones which we cannot handle anyway
            //
            // 0: https://github.com/apple-oss-distributions/libpthread/blob/67e155c94093be9a204b69637d198eceff2c7c46/src/qos.c#L171-L177
            let errno = unsafe { *libc::__error() };
            unreachable!("`pthread_get_qos_class_np` failed unexpectedly (os error {errno})");
        }

        match qos_class_raw {
            libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE => Some(QoSClass::UserInteractive),
            libc::qos_class_t::QOS_CLASS_USER_INITIATED => Some(QoSClass::UserInitiated),
            libc::qos_class_t::QOS_CLASS_DEFAULT => None, // QoS has never been set
            libc::qos_class_t::QOS_CLASS_UTILITY => Some(QoSClass::Utility),
            libc::qos_class_t::QOS_CLASS_BACKGROUND => Some(QoSClass::Background),

            libc::qos_class_t::QOS_CLASS_UNSPECIFIED => {
                // Using manual scheduling APIs causes threads to “opt out” of QoS.
                // At this point they become incompatible with QoS,
                // and as such have the “unspecified” QoS class.
                //
                // Panic instead of returning an error
                // to maintain the invariant that we only use QoS APIs.
                panic!("tried to get QoS of thread which has opted out of QoS")
            }
        }
    }
}

// FIXME: Windows has QoS APIs, we should use them!
#[cfg(not(target_vendor = "apple"))]
mod imp {
    use super::QoSClass;

    pub(super) const IS_QOS_AVAILABLE: bool = false;

    pub(super) fn set_current_thread_qos_class(_: QoSClass) {}

    pub(super) fn get_current_thread_qos_class() -> Option<QoSClass> {
        None
    }
}
