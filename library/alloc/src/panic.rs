use crate::rc::Rc;
use crate::sync::Arc;
use core::panic::{RefUnwindSafe, UnwindSafe};

// not covered via the Shared impl above b/c the inner contents use
// Cell/AtomicUsize, but the usage here is unwind safe so we can lift the
// impl up one level to Arc/Rc itself
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Rc<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Arc<T> {}
