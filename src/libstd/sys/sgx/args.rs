use ffi::OsString;
use super::abi::usercalls::{alloc, raw::ByteBuffer};
use sync::atomic::{AtomicUsize, Ordering};
use sys::os_str::Buf;
use sys_common::FromInner;
use slice;

static ARGS: AtomicUsize = AtomicUsize::new(0);
type ArgsStore = Vec<OsString>;

pub unsafe fn init(argc: isize, argv: *const *const u8) {
    if argc != 0 {
        let args = alloc::User::<[ByteBuffer]>::from_raw_parts(argv as _, argc as _);
        let args = args.iter()
            .map( |a| OsString::from_inner(Buf { inner: a.copy_user_buffer() }) )
            .collect::<ArgsStore>();
        ARGS.store(Box::into_raw(Box::new(args)) as _, Ordering::Relaxed);
    }
}

pub unsafe fn cleanup() {
    let args = ARGS.swap(0, Ordering::Relaxed);
    if args != 0 {
        drop(Box::<ArgsStore>::from_raw(args as _))
    }
}

pub fn args() -> Args {
    let args = unsafe { (ARGS.load(Ordering::Relaxed) as *const ArgsStore).as_ref() };
    if let Some(args) = args {
        Args(args.iter())
    } else {
        Args([].iter())
    }
}

pub struct Args(slice::Iter<'static, OsString>);

impl Args {
    pub fn inner_debug(&self) -> &[OsString] {
        self.0.as_slice()
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
