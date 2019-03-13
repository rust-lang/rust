//! Extensions to `std::thread` for Windows.

#![stable(feature = "thread_extensions", since = "1.9.0")]

use crate::os::windows::io::{RawHandle, AsRawHandle, IntoRawHandle};
use crate::thread;
use crate::sys_common::{AsInner, IntoInner};

#[stable(feature = "thread_extensions", since = "1.9.0")]
impl<T> AsRawHandle for thread::JoinHandle<T> {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as *mut _
    }
}

#[stable(feature = "thread_extensions", since = "1.9.0")]
impl<T> IntoRawHandle for thread::JoinHandle<T>  {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}
