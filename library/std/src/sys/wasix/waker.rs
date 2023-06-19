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

fn _waker_register(cx: &mut crate::task::Context<'_>) -> RawWasiWakerRef {
    let waker = Box::new(RawWasiWaker {
        inner: cx.waker().clone()
    });
    let waker = Box::into_raw(waker) as *mut RawWasiWaker;
    let waker = waker as usize;
    (waker as u64).into()
}

pub(crate) struct RawWasiWakerRef {
    id: Option<u64>,
}
impl From<u64>
for RawWasiWakerRef {
    fn from(val: u64) -> Self {
        Self {
            id: Some(val)
        }
    }
}
impl Drop
for RawWasiWakerRef {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            _waker_drop(id);
        }
    }
}
impl Into<u64>
for RawWasiWakerRef {
    fn into(mut self) -> u64 {
        self.id.take().unwrap()
    }
}

pub(crate) fn asyncify<T, F>(cx: &mut Context<'_>, funct: F) -> Poll<io::Result<T>>
where F: FnOnce(RawWasiWakerRef) -> Result<T, Errno> {
    let waker_id = _waker_register(cx);
    let ret = funct(waker_id);
    match ret {
        Ok(ret) => {
            Poll::Ready(Ok(ret))
        },
        Err(wasi::ERRNO_PENDING) => {
            Poll::Pending
        },
        Err(err) => {
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
