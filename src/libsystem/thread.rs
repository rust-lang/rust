pub use imp::thread as imp;

pub mod traits {
    pub use super::Thread as sys_Thread;
}

pub mod prelude {
    pub use super::imp::Thread;
    pub use super::traits::*;
    //pub use super::info;
}

use error::prelude::*;
use core::time::Duration;

pub trait Thread {
    unsafe fn new(stack: usize, f: unsafe extern fn(usize) -> usize, data: usize) -> Result<Self> where Self: Sized;
    //unsafe fn named(name: Option<&str>) -> Self where Self: Sized;

    fn join(self) -> Result<()>;

    fn set_name(name: &str) -> Result<()>;
    fn yield_();
    fn sleep(dur: Duration) -> Result<()>;
}

/*pub mod info {
    use core::cell::RefCell;
    use collections::String;
    use alloc::arc::Arc;
    use super::imp::Thread;
    //use thread::LocalKeyState;

    struct ThreadInfo {
        stack_guard: Option<usize>,
        thread: Arc<Thread>,
    }

    //thread_local! { static THREAD_INFO: RefCell<Option<ThreadInfo>> = RefCell::new(None) }

    impl ThreadInfo {
        fn with<R, F>(f: F) -> Option<R> where F: FnOnce(&mut ThreadInfo) -> R {
            /*if THREAD_INFO.state() == LocalKeyState::Destroyed {
                return None
            }

            THREAD_INFO.with(move |c| {
                if c.borrow().is_none() {
                    *c.borrow_mut() = Some(ThreadInfo {
                        stack_guard: None,
                        thread: Thread::named(None),
                    })
                }
                Some(f(c.borrow_mut().as_mut().unwrap()))
            })*/

            panic!()
        }
    }

    pub fn current_thread() -> Option<Arc<Thread>> {
        ThreadInfo::with(|info| info.thread.clone())
    }

    pub fn stack_guard() -> Option<usize> {
        ThreadInfo::with(|info| info.stack_guard).and_then(|o| o)
    }

    pub fn set(stack_guard: Option<usize>, thread: Arc<Thread>) {
        panic!()
        /*THREAD_INFO.with(|c| assert!(c.borrow().is_none()));
        THREAD_INFO.with(move |c| *c.borrow_mut() = Some(ThreadInfo{
            stack_guard: stack_guard,
            thread: thread,
        }));*/
    }
}*/
