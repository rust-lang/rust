use crate::task::{Poll, Context, Waker};
use crate::io;
use crate::mem::ManuallyDrop;
use wasi::x::Errno;

// Callback when a waker is woken
#[no_mangle]
#[allow(fuzzy_provenance_casts)]
pub extern "C" fn _wake(waker0: i64, waker1: i64, waker2: i64, waker3: i64, waker4: i64, waker5: i64, waker6: i64, waker7: i64) {
    let process = |waker: i64| {
        if waker > 0 {
            unsafe {
                let waker = waker.abs() as *mut Waker;
                ManuallyDrop::new(Box::from_raw(waker))
            }.wake_by_ref();
        } else if waker < 0 {
            drop(unsafe {
                let waker = waker.abs() as *mut Waker;
                Box::from_raw(waker)
            })
        }
    };
    process(waker0);
    process(waker1);
    process(waker2);
    process(waker3);
    process(waker4);
    process(waker5);
    process(waker6);
    process(waker7);
}

pub fn waker_register(cx: &mut crate::task::Context<'_>) -> RawWakerRef {
    let waker = Box::new(cx.waker().clone());
    let waker = Box::into_raw(waker) as *mut Waker;
    RawWakerRef {
        raw: Some(waker as u64)
    }
}

pub struct RawWakerRef {
    raw: Option<u64>,
}
impl Drop
for RawWakerRef {
    #[allow(fuzzy_provenance_casts)]
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            drop(unsafe {
                let waker = raw as *mut Waker;
                Box::from_raw(waker)
            })
        }
    }
}
impl Into<u64>
for RawWakerRef {
    fn into(mut self) -> u64 {
        self.raw.take().unwrap()
    }
}

pub(crate) fn asyncify<T, F>(cx: &mut Context<'_>, funct: F) -> Poll<io::Result<T>>
where F: FnOnce(RawWakerRef) -> Result<T, Errno> {
    let waker_id = waker_register(cx);
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
