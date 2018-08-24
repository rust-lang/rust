use super::mutex::Mutex;

pub struct RWLock {
    mutex: Mutex
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            mutex: Mutex::new()
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        self.mutex.lock();
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        self.mutex.try_lock()
    }

    #[inline]
    pub unsafe fn write(&self) {
        self.mutex.lock();
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.mutex.try_lock()
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        self.mutex.unlock();
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        self.mutex.unlock();
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        self.mutex.destroy();
    }
}
