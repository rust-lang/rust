//! hi
#![unstable(feature = "core_backtrace", issue = "74465")]
use crate::fmt;

// perma(?)-unstable
#[unstable(feature = "core_backtrace", issue = "74465")]
///
pub trait RawBacktraceImpl: fmt::Debug + fmt::Display + 'static {
    ///
    unsafe fn drop_and_free(self: *mut Self);
}

#[unstable(feature = "core_backtrace", issue = "74465")]
///
pub struct Backtrace {
    inner: *mut dyn RawBacktraceImpl,
}

#[unstable(feature = "core_backtrace", issue = "74465")]
unsafe impl Send for Backtrace {}

#[unstable(feature = "core_backtrace", issue = "74465")]
unsafe impl Sync for Backtrace {}

#[unstable(feature = "core_backtrace", issue = "74465")]
impl Drop for Backtrace {
    fn drop(&mut self) {
        unsafe { RawBacktraceImpl::drop_and_free(self.inner) }
    }
}

#[unstable(feature = "core_backtrace", issue = "74465")]
impl fmt::Debug for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let imp: &dyn RawBacktraceImpl = unsafe { &*self.inner };
        fmt::Debug::fmt(imp, fmt)
    }
}

#[unstable(feature = "core_backtrace", issue = "74465")]
impl fmt::Display for Backtrace {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let imp: &dyn RawBacktraceImpl = unsafe { &*self.inner };
        fmt::Display::fmt(imp, fmt)
    }
}