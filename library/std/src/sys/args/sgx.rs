use crate::ffi::OsString;
use crate::sync::OnceLock;
use crate::sys::os_str::Buf;
use crate::sys::pal::abi::usercalls::alloc;
use crate::sys::pal::abi::usercalls::raw::ByteBuffer;
use crate::sys_common::FromInner;
use crate::{fmt, slice};

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3sgx4args4ARGSE")]
static ARGS: OnceLock<ArgsStore> = OnceLock::new();
type ArgsStore = Vec<OsString>;

#[cfg_attr(test, allow(dead_code))]
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    if argc != 0 {
        ARGS.get_or_init(|| {
            let args = unsafe { alloc::User::<[ByteBuffer]>::from_raw_parts(argv as _, argc as _) };
            args.iter()
                .map(|a| OsString::from_inner(Buf { inner: a.copy_user_buffer() }))
                .collect::<ArgsStore>()
        });
    }
}

pub fn args() -> Args {
    let args = ARGS.get();
    if let Some(args) = args { Args(args.iter()) } else { Args([].iter()) }
}

pub struct Args(slice::Iter<'static, OsString>);

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.0.next().cloned()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.0.next_back().cloned()
    }
}
