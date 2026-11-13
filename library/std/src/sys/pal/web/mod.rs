//! System bindings for the web platforms.

use crate::io;

#[path = "../wasi/conf.rs"]
pub mod conf;
#[path = "../unix/time.rs"]
pub mod time;

#[path = "../wasi/cabi_realloc.rs"]
mod cabi_realloc;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[expect(dead_code)]
mod common;
pub use common::{cleanup, init, unsupported};

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(io::Error::last_os_error()) } else { Ok(t) }
}

pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.is_interrupted() => {}
            other => return other,
        }
    }
}
