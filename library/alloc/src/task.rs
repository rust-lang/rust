#![unstable(feature = "wake_trait", issue = "69912")]
//! Types and Traits for working with asynchronous tasks.
use core::mem::ManuallyDrop;
use core::task::{RawWaker, RawWakerVTable, Waker};

use crate::sync::Arc;

/// The implementation of waking a task on an executor.
///
/// This trait can be used to create a [`Waker`]. An executor can define an
/// implementation of this trait, and use that to construct a Waker to pass
/// to the tasks that are executed on that executor.
///
/// This trait is a memory-safe and ergonomic alternative to constructing a
/// [`RawWaker`]. It supports the common executor design in which the data used
/// to wake up a task is stored in an [`Arc`]. Some executors (especially
/// those for embedded systems) cannot use this API, which is why [`RawWaker`]
/// exists as an alternative for those systems.
#[unstable(feature = "wake_trait", issue = "69912")]
pub trait Wake {
    /// Wake this task.
    #[unstable(feature = "wake_trait", issue = "69912")]
    fn wake(self: Arc<Self>);

    /// Wake this task without consuming the waker.
    ///
    /// If an executor supports a cheaper way to wake without consuming the
    /// waker, it should override this method. By default, it clones the
    /// [`Arc`] and calls `wake` on the clone.
    #[unstable(feature = "wake_trait", issue = "69912")]
    fn wake_by_ref(self: &Arc<Self>) {
        self.clone().wake();
    }
}

#[cfg_attr(bootstrap, allow(rustc::ineffective_unstable_trait_impl))]
#[cfg_attr(not(bootstrap), allow(ineffective_unstable_trait_impl))]
#[unstable(feature = "wake_trait", issue = "69912")]
impl<W: Wake + Send + Sync + 'static> From<Arc<W>> for Waker {
    fn from(waker: Arc<W>) -> Waker {
        // SAFETY: This is safe because raw_waker safely constructs
        // a RawWaker from Arc<W>.
        unsafe { Waker::from_raw(raw_waker(waker)) }
    }
}

#[cfg_attr(bootstrap, allow(rustc::ineffective_unstable_trait_impl))]
#[cfg_attr(not(bootstrap), allow(ineffective_unstable_trait_impl))]
#[unstable(feature = "wake_trait", issue = "69912")]
impl<W: Wake + Send + Sync + 'static> From<Arc<W>> for RawWaker {
    fn from(waker: Arc<W>) -> RawWaker {
        raw_waker(waker)
    }
}

// NB: This private function for constructing a RawWaker is used, rather than
// inlining this into the `From<Arc<W>> for RawWaker` impl, to ensure that
// the safety of `From<Arc<W>> for Waker` does not depend on the correct
// trait dispatch - instead both impls call this function directly and
// explicitly.
#[inline(always)]
fn raw_waker<W: Wake + Send + Sync + 'static>(waker: Arc<W>) -> RawWaker {
    // Increment the reference count of the arc to clone it.
    unsafe fn clone_waker<W: Wake + Send + Sync + 'static>(waker: *const ()) -> RawWaker {
        unsafe { Arc::incr_strong_count(waker as *const W) };
        RawWaker::new(
            waker as *const (),
            &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
        )
    }

    // Wake by value, moving the Arc into the Wake::wake function
    unsafe fn wake<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { Arc::from_raw(waker as *const W) };
        <W as Wake>::wake(waker);
    }

    // Wake by reference, wrap the waker in ManuallyDrop to avoid dropping it
    unsafe fn wake_by_ref<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { ManuallyDrop::new(Arc::from_raw(waker as *const W)) };
        <W as Wake>::wake_by_ref(&waker);
    }

    // Decrement the reference count of the Arc on drop
    unsafe fn drop_waker<W: Wake + Send + Sync + 'static>(waker: *const ()) {
        unsafe { Arc::decr_strong_count(waker as *const W) };
    }

    RawWaker::new(
        Arc::into_raw(waker) as *const (),
        &RawWakerVTable::new(clone_waker::<W>, wake::<W>, wake_by_ref::<W>, drop_waker::<W>),
    )
}
