use crate::sys::hermit::abi;

pub struct Mutex {
    inner: abi::mutex::Mutex,
}

pub type MovableMutex = Mutex;

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: abi::mutex::Mutex::new() }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        self.inner.init();
    }

    #[inline]
    pub unsafe fn lock(&self) {
        self.inner.lock();
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        self.inner.unlock();
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        self.inner.try_lock()
    }
}
