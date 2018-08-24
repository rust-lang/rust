use boxed::FnBox;
use ffi::CStr;
use io;
use sys::{unsupported, Void};
use time::Duration;

pub struct Thread(Void);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    pub unsafe fn new<'a>(_stack: usize, _p: Box<dyn FnBox() + 'a>)
        -> io::Result<Thread>
    {
        unsupported()
    }

    pub fn yield_now() {
        // do nothing
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(_dur: Duration) {
        panic!("can't sleep");
    }

    pub fn join(self) {
        match self.0 {}
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
    pub unsafe fn deinit() {}
}
