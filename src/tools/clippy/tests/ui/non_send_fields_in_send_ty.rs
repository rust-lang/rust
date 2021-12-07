#![warn(clippy::non_send_fields_in_send_ty)]
#![feature(extern_types)]

use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};

// disrustor / RUSTSEC-2020-0150
pub struct RingBuffer<T> {
    data: Vec<UnsafeCell<T>>,
    capacity: usize,
    mask: usize,
}

unsafe impl<T> Send for RingBuffer<T> {}

// noise_search / RUSTSEC-2020-0141
pub struct MvccRwLock<T> {
    raw: *const T,
    lock: Mutex<Box<T>>,
}

unsafe impl<T> Send for MvccRwLock<T> {}

// async-coap / RUSTSEC-2020-0124
pub struct ArcGuard<RC, T> {
    inner: T,
    head: Arc<RC>,
}

unsafe impl<RC, T: Send> Send for ArcGuard<RC, T> {}

// rusb / RUSTSEC-2020-0098
extern "C" {
    type libusb_device_handle;
}

pub trait UsbContext {
    // some user trait that does not guarantee `Send`
}

pub struct DeviceHandle<T: UsbContext> {
    context: T,
    handle: NonNull<libusb_device_handle>,
}

unsafe impl<T: UsbContext> Send for DeviceHandle<T> {}

// Other basic tests
pub struct NoGeneric {
    rc_is_not_send: Rc<String>,
}

unsafe impl Send for NoGeneric {}

pub struct MultiField<T> {
    field1: T,
    field2: T,
    field3: T,
}

unsafe impl<T> Send for MultiField<T> {}

pub enum MyOption<T> {
    MySome(T),
    MyNone,
}

unsafe impl<T> Send for MyOption<T> {}

// Test types that contain `NonNull` instead of raw pointers (#8045)
pub struct WrappedNonNull(UnsafeCell<NonNull<()>>);

unsafe impl Send for WrappedNonNull {}

// Multiple type parameters
pub struct MultiParam<A, B> {
    vec: Vec<(A, B)>,
}

unsafe impl<A, B> Send for MultiParam<A, B> {}

// Tests for raw pointer heuristic
extern "C" {
    type NonSend;
}

pub struct HeuristicTest {
    // raw pointers are allowed
    field1: Vec<*const NonSend>,
    field2: [*const NonSend; 3],
    field3: (*const NonSend, *const NonSend, *const NonSend),
    // not allowed when it contains concrete `!Send` field
    field4: (*const NonSend, Rc<u8>),
    // nested raw pointer is also allowed
    field5: Vec<Vec<*const NonSend>>,
}

unsafe impl Send for HeuristicTest {}

// Test attributes
#[allow(clippy::non_send_fields_in_send_ty)]
pub struct AttrTest1<T>(T);

pub struct AttrTest2<T> {
    #[allow(clippy::non_send_fields_in_send_ty)]
    field: T,
}

pub enum AttrTest3<T> {
    #[allow(clippy::non_send_fields_in_send_ty)]
    Enum1(T),
    Enum2(T),
}

unsafe impl<T> Send for AttrTest1<T> {}
unsafe impl<T> Send for AttrTest2<T> {}
unsafe impl<T> Send for AttrTest3<T> {}

// Multiple non-overlapping `Send` for a single type
pub struct Complex<A, B> {
    field1: A,
    field2: B,
}

unsafe impl<P> Send for Complex<P, u32> {}

// `MutexGuard` is non-Send
unsafe impl<Q: Send> Send for Complex<Q, MutexGuard<'static, bool>> {}

fn main() {}
