//! An opaque façade around platform-specific QoS APIs.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
// Please maintain order from least to most priority for the derived `Ord` impl.
pub enum ThreadIntent {
    /// Any thread which does work that isn’t in the critical path of the user typing
    /// (e.g. processing Go To Definition).
    Worker,

    /// Any thread which does work caused by the user typing
    /// (e.g. processing syntax highlighting).
    LatencySensitive,
}

impl ThreadIntent {
    // These APIs must remain private;
    // we only want consumers to set thread intent
    // either during thread creation or using our pool impl.

    pub(super) fn apply_to_current_thread(self) {
        let class = thread_intent_to_qos_class(self);
        set_current_thread_qos_class(class);
    }

    pub(super) fn assert_is_used_on_current_thread(self) {
        if IS_QOS_AVAILABLE {
            let class = thread_intent_to_qos_class(self);
            assert_eq!(get_current_thread_qos_class(), Some(class));
        }
    }
}

use imp::QoSClass;

const IS_QOS_AVAILABLE: bool = imp::IS_QOS_AVAILABLE;

fn set_current_thread_qos_class(class: QoSClass) {
    imp::set_current_thread_qos_class(class)
}

fn get_current_thread_qos_class() -> Option<QoSClass> {
    imp::get_current_thread_qos_class()
}

fn thread_intent_to_qos_class(intent: ThreadIntent) -> QoSClass {
    imp::thread_intent_to_qos_class(intent)
}

// All Apple platforms use XNU as their kernel
// and thus have the concept of QoS.
#[cfg(target_vendor = "apple")]
mod imp {
    use super::ThreadIntent;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    // Please maintain order from least to most priority for the derived `Ord` impl.
    pub(super) enum QoSClass {
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

    pub(super) fn thread_intent_to_qos_class(intent: ThreadIntent) -> QoSClass {
        match intent {
            ThreadIntent::Worker => QoSClass::Utility,
            ThreadIntent::LatencySensitive => QoSClass::UserInitiated,
        }
    }
}

// FIXME: Windows has QoS APIs, we should use them!
#[cfg(not(target_vendor = "apple"))]
mod imp {
    use super::ThreadIntent;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub(super) enum QoSClass {
        Default,
    }

    pub(super) const IS_QOS_AVAILABLE: bool = false;

    pub(super) fn set_current_thread_qos_class(_: QoSClass) {}

    pub(super) fn get_current_thread_qos_class() -> Option<QoSClass> {
        None
    }

    pub(super) fn thread_intent_to_qos_class(_: ThreadIntent) -> QoSClass {
        QoSClass::Default
    }
}
