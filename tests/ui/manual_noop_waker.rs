#![warn(clippy::manual_noop_waker)]
use std::sync::Arc;
use std::task::Wake;

struct PartialWaker;
impl Wake for PartialWaker {
    //~^ ERROR: manual implementation of a no-op waker
    fn wake(self: Arc<Self>) {}
}

struct MyWakerPartial;
impl Wake for MyWakerPartial {
    //~^ manual_noop_waker
    fn wake(self: Arc<Self>) {}
    // wake_by_ref not implemented, uses default
}

trait CustomWake {
    fn wake(self);
}

impl CustomWake for () {
    fn wake(self) {}
}

mod custom_module {
    use std::sync::Arc;

    // Custom Wake trait that should NOT trigger the lint
    pub trait Wake {
        fn wake(self: Arc<Self>);
        fn wake_by_ref(self: &Arc<Self>);
    }

    pub struct CustomWaker;
    impl Wake for CustomWaker {
        fn wake(self: Arc<Self>) {}
        fn wake_by_ref(self: &Arc<Self>) {}
    }
}
