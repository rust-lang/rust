use crate::task::{Poll, Context, Waker};
use crate::io;
use wasi::x::Errno;

// Callback when a waker is woken
#[no_mangle]
#[allow(fuzzy_provenance_casts)]
pub extern "C" fn _waker_wake(waker: u64) {
    let waker = waker as usize;
    let waker = unsafe {
        let waker = waker as *mut RawWasiWaker;
        Box::from_raw(waker)
    };
    waker.wake();
}

// Callback when a waker is dropped
#[no_mangle]
#[allow(fuzzy_provenance_casts)]
pub extern "C" fn _waker_drop(waker: u64) {
    let waker = waker as usize;
    let waker = unsafe {
        let waker = waker as *mut RawWasiWaker;
        Box::from_raw(waker)
    };
    drop(waker);
}

fn _waker_register(cx: &mut crate::task::Context<'_>) -> u64 {
    let waker = Box::new(RawWasiWaker {
        inner: cx.waker().clone()
    });
    let waker = Box::into_raw(waker) as *mut RawWasiWaker;
    let waker = waker as usize;
    waker as u64
}

pub(crate) fn asyncify<T, F>(cx: &mut Context<'_>, funct: F) -> Poll<io::Result<T>>
where F: FnOnce(u64) -> Result<T, Errno> {
    let waker_id = _waker_register(cx);
    let ret = funct(waker_id);
    match ret {
        Ok(ret) => {
            _waker_drop(waker_id);
            Poll::Ready(Ok(ret))
        },
        Err(wasi::ERRNO_PENDING) => {
            Poll::Pending
        },
        Err(err) => {
            _waker_drop(waker_id);
            Poll::Ready(Err(super::err2io(err)))
        }
    }
}

struct RawWasiWaker {
    inner: Waker,
}

impl RawWasiWaker {
    pub fn wake(self) {
        self.inner.wake();
    }
}
