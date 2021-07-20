use super::abi::usercalls::{alloc, raw::ByteBuffer};
use crate::ffi::OsString;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::os_str::Buf;
use crate::sys_common::FromInner;
use crate::vec::IntoIter;

#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx4args4ARGSE"]
static ARGS: AtomicUsize = AtomicUsize::new(0);
type ArgsStore = Vec<OsString>;

#[cfg_attr(test, allow(dead_code))]
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    if argc != 0 {
        let args = unsafe { alloc::User::<[ByteBuffer]>::from_raw_parts(argv as _, argc as _) };
        let args = args
            .iter()
            .map(|a| OsString::from_inner(Buf { inner: a.copy_user_buffer() }))
            .collect::<ArgsStore>();
        ARGS.store(Box::into_raw(Box::new(args)) as _, Ordering::Relaxed);
    }
}

pub fn args() -> Args {
    let args = unsafe { (ARGS.load(Ordering::Relaxed) as *const ArgsStore).as_ref() };
    if let Some(args) = args { args.clone().into_iter() } else { Vec::new().into_iter() }
}

pub type Args = IntoIter<OsString>;
