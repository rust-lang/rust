//@ edition:2018

use std::sync::Arc;
use std::task::{
    Waker, RawWaker, RawWakerVTable,
};

macro_rules! waker_vtable {
    ($ty:ident) => {
        &RawWakerVTable::new(
            clone_arc_raw::<$ty>,
            wake_arc_raw::<$ty>,
            wake_by_ref_arc_raw::<$ty>,
            drop_arc_raw::<$ty>,
        )
    };
}

pub trait ArcWake {
    fn wake(self: Arc<Self>);

    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.clone().wake()
    }

    fn into_waker(wake: Arc<Self>) -> Waker where Self: Sized
    {
        let ptr = Arc::into_raw(wake) as *const ();

        unsafe {
            Waker::from_raw(RawWaker::new(ptr, waker_vtable!(Self)))
        }
    }
}

unsafe fn increase_refcount<T: ArcWake>(data: *const ()) {
    // Retain Arc by creating a copy
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    let arc_clone = arc.clone();
    // Forget the Arcs again, so that the refcount isn't decrased
    let _ = Arc::into_raw(arc);
    let _ = Arc::into_raw(arc_clone);
}

unsafe fn clone_arc_raw<T: ArcWake>(data: *const ()) -> RawWaker {
    increase_refcount::<T>(data);
    RawWaker::new(data, waker_vtable!(T))
}

unsafe fn drop_arc_raw<T: ArcWake>(data: *const ()) {
    // Drop Arc
    let _: Arc<T> = Arc::from_raw(data as *const T);
}

unsafe fn wake_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake(arc);
}

unsafe fn wake_by_ref_arc_raw<T: ArcWake>(data: *const ()) {
    let arc: Arc<T> = Arc::from_raw(data as *const T);
    ArcWake::wake_by_ref(&arc);
    let _ = Arc::into_raw(arc);
}
