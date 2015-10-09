pub use imp::sync as imp;

pub mod traits {
    pub use sync::{Sync as sys_Sync, Condvar as sys_Condvar, RwLock as sys_RwLock, ReentrantMutex as sys_Remutex, Mutex as sys_Mutex, Lock as sys_Lock, Once as sys_Once};
}

pub mod prelude {
    pub use super::imp::{Sync, Mutex, ReentrantMutex, RwLock, Condvar, Once};
    pub use super::traits::*;
    pub use super::LockGuard;

    /*pub type Mutex = <Sync as sys_Sync>::Mutex;
    pub type Condvar = <Sync as sys_Sync>::Condvar;
    pub type RwLock = <Sync as sys_Sync>::RwLock;
    pub type ReentrantMutex = <Sync as sys_Sync>::ReentrantMutex;
    pub type Once = <Sync as sys_Sync>::Once;*/
}

use core::time;
use core::marker;

pub trait Lock {
    unsafe fn lock(&self);
    unsafe fn unlock(&self);
    unsafe fn try_lock(&self) -> bool;
    unsafe fn destroy(&self);
}

pub trait Mutex: Lock + marker::Sync + marker::Send {
    //const fn new() -> Self;
}

pub trait ReentrantMutex: Lock + marker::Sync + marker::Send {
    //const fn uninitialized() -> Self;

    unsafe fn init(&mut self);
}

pub trait RwLock: marker::Sync + marker::Send {
    //const fn new() -> Self;

    unsafe fn read(&self);
    unsafe fn try_read(&self) -> bool;
    unsafe fn write(&self);
    unsafe fn try_write(&self) -> bool;
    unsafe fn read_unlock(&self);
    unsafe fn write_unlock(&self);
    unsafe fn destroy(&self);
}

pub trait Condvar: marker::Sync + marker::Send {
    type Mutex: Mutex;

    //const fn new() -> Self;

    unsafe fn notify_one(&self);
    unsafe fn notify_all(&self);
    unsafe fn wait(&self, mutex: &Self::Mutex);
    unsafe fn wait_timeout(&self, mutex: &Self::Mutex, dur: time::Duration) -> bool;
    unsafe fn destroy(&self);
}

pub trait Sync {
    type Mutex: Mutex;
    type ReentrantMutex: ReentrantMutex;
    type RwLock: RwLock;
    type Condvar: Condvar<Mutex=Self::Mutex>;
    type Once: Once;
}

pub trait Once: marker::Sync + marker::Send {
    //const fn new() -> Self;

    fn call_once<F: FnOnce()>(&'static self, f: F);
}

pub struct LockGuard<'a, L: Lock + 'a>(&'a L);
impl<'a, L: Lock + 'a> Drop for LockGuard<'a, L> {
    fn drop(&mut self) {
        unsafe { self.0.unlock(); }
    }
}
