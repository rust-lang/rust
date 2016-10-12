use cell::Cell;

pub struct RWLock {
    write_locked: Cell<bool>,
    num_readers: Cell<usize>,
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            write_locked: Cell::new(false),
            num_readers: Cell::new(0),
        }
    }
    #[inline]
    pub unsafe fn read(&self) {
        debug_assert!(!self.write_locked.get());
        self.num_readers.set(self.num_readers.get()+1)
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let already_locked=self.write_locked.get();
        if !already_locked {
            self.read()
        }
        !already_locked
    }
    #[inline]
    pub unsafe fn write(&self) {
        debug_assert!(!self.write_locked.get());
        debug_assert_eq!(self.num_readers.get(),0);
        self.write_locked.set(true)
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let already_locked=self.write_locked.get() || self.num_readers.get()==0;
        if !already_locked {
            self.write()
        }
        !already_locked
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        debug_assert!(self.num_readers.get()>0);
        self.num_readers.set(self.num_readers.get()-1);
    }
    #[inline]
    pub unsafe fn write_unlock(&self) {
        debug_assert!(self.write_locked.get());
        self.write_locked.set(false)
    }
    #[inline]
    pub unsafe fn destroy(&self) {
        self.write()
    }
}
