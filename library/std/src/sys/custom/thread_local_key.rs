use crate::os::custom::thread::{ThreadId, IMPL};
use crate::sync::{Mutex, RwLock};

pub type Key = usize;

const NULL: *mut u8 = core::ptr::null_mut();

type Destructor = Option<unsafe extern "C" fn(*mut u8)>;

fn thread_id() -> Option<ThreadId> {
    let reader = IMPL.read_no_poison_check();
    reader.as_ref()?.current_thread_id()
}

// safety: each ThreadStorage is virtually owned by one thread
unsafe impl Sync for ThreadStorage {}
unsafe impl Send for ThreadStorage {}

struct ThreadStorage(Mutex<Vec<*mut u8>>);

impl ThreadStorage {
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    pub fn get(&self, key: Key) -> *mut u8 {
        let locked = self.0.lock_no_poison_check();
        match locked.get(key - 1) {
            Some(pointer_ref) => *pointer_ref,
            None => NULL,
        }
    }

    pub fn set(&self, key: Key, value: *mut u8) {
        let mut locked = self.0.lock_no_poison_check();
        locked.resize(key, NULL);
        locked[key - 1] = value;
    }
}

struct Slots {
    destructors: RwLock<Vec<Destructor>>,
    initial_thread: ThreadStorage,
    other_threads: RwLock<Vec<(ThreadId, ThreadStorage)>>,
}

impl Slots {
    pub const fn new() -> Self {
        Self {
            destructors: RwLock::new(Vec::new()),
            initial_thread: ThreadStorage::new(),
            other_threads: RwLock::new(Vec::new()),
        }
    }

    pub fn push(&self, destructor: Destructor) -> Key {
        let mut destructors = self.destructors.write_no_poison_check();
        destructors.push(destructor);
        destructors.len()
    }

    pub fn get_dtor(&self, key: Key) -> Destructor {
        let destructors = self.destructors.read_no_poison_check();
        destructors[key - 1]
    }

    pub fn with_thread_storage<T, F: FnOnce(&ThreadStorage) -> T>(&self, callback: F) -> T {
        if let Some(id) = thread_id() {
            loop {
                let finder = |(tid, _): &(ThreadId, ThreadStorage)| tid.cmp(&id);

                {
                    let other_threads = self.other_threads.read_no_poison_check();
                    let position = other_threads.binary_search_by(finder);

                    if let Ok(i) = position {
                        return callback(&other_threads[i].1);
                    }
                }

                // cannot re-use `position` (race condition)
                let mut other_threads = self.other_threads.write_no_poison_check();
                let i = other_threads.binary_search_by(finder).unwrap_err();
                other_threads.insert(i, (id, ThreadStorage::new()));
            }
        } else {
            callback(&self.initial_thread)
        }
    }
}

static SLOTS: Slots = Slots::new();

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    SLOTS.push(dtor)
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    SLOTS.with_thread_storage(|storage| storage.set(key, value))
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    SLOTS.with_thread_storage(|storage| storage.get(key))
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let value = get(key);
    if !value.is_null() {
        if let Some(destructor) = SLOTS.get_dtor(key) {
            destructor(value)
        }
    }
}
