use crate::task::{Poll, Context, Waker};
use crate::io;
use wasi::x::Errno;

// Callback when a waker is woken
#[no_mangle]
#[allow(fuzzy_provenance_casts)]
pub extern "C" fn _wake(waker0: i64, waker1: i64, waker2: i64, waker3: i64, waker4: i64, waker5: i64, waker6: i64, waker7: i64) {
    let process = |waker: i64| {
        if waker > 0 {
            unsafe {
                let waker = waker.abs() as *mut Waker;
                Box::from_raw(waker)
            }.wake();
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

#[allow(fuzzy_provenance_casts)]
fn waker_register(cx: &mut crate::task::Context<'_>) -> (RawWakerRef, u64) {
    let waker = Box::new(cx.waker().clone());
    let waker = Box::into_raw(waker) as *mut Waker;
    let waker_id = waker as u64;
    (
        RawWakerRef {
            raw: waker_id
        },
        waker_id
    )
}

pub struct RawWakerRef {
    raw: u64
}
impl Into<u64>
for RawWakerRef {
    fn into(self) -> u64 {
        self.raw
    }
}

#[allow(fuzzy_provenance_casts)]
pub(crate) fn asyncify<T, F>(cx: &mut Context<'_>, funct: F) -> Poll<io::Result<T>>
where F: FnOnce(RawWakerRef) -> Result<T, Errno> {
    let (waker_ref, waker_id) = waker_register(cx);
    let ret = funct(waker_ref);
    match ret {
        Ok(ret) => {
            drop(unsafe {
                let waker = waker_id as *mut Waker;
                Box::from_raw(waker)
            });
            Poll::Ready(Ok(ret))
        },
        Err(wasi::ERRNO_PENDING) => {
            Poll::Pending
        },
        Err(err) => {
            drop(unsafe {
                let waker = waker_id as *mut Waker;
                Box::from_raw(waker)
            });
            Poll::Ready(Err(super::err2io(err)))
        }
    }
}
