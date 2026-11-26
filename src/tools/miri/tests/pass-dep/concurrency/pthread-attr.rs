//@ignore-target: windows # No pthreads on Windows
//@run-native

use std::mem::MaybeUninit;
use std::ptr;

extern "C" fn thread_start(_arg: *mut libc::c_void) -> *mut libc::c_void {
    ptr::null_mut()
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE).try_into().unwrap() }
}

/// A valid stack size: at least `PTHREAD_STACK_MIN` and page-aligned.
fn normal_stack_size() -> usize {
    libc::PTHREAD_STACK_MIN.next_multiple_of(page_size())
}

/// Spawns a thread with the given attributes, and joins it if that succeeded.
fn spawn(attr: *const libc::pthread_attr_t) -> i32 {
    unsafe {
        let mut thread = MaybeUninit::<libc::pthread_t>::uninit();
        let ret = libc::pthread_create(thread.as_mut_ptr(), attr, thread_start, ptr::null_mut());
        if ret == 0 {
            assert_eq!(libc::pthread_join(thread.assume_init(), ptr::null_mut()), 0);
        }
        ret
    }
}

fn test_init_destroy() {
    unsafe {
        let mut attr = MaybeUninit::<libc::pthread_attr_t>::uninit();
        assert_eq!(libc::pthread_attr_init(attr.as_mut_ptr()), 0);
        assert_eq!(libc::pthread_attr_destroy(attr.as_mut_ptr()), 0);
        // It can be initialized again after being destroyed.
        assert_eq!(libc::pthread_attr_init(attr.as_mut_ptr()), 0);
        assert_eq!(libc::pthread_attr_destroy(attr.as_mut_ptr()), 0);
    }
}

fn test_setstacksize() {
    unsafe {
        let mut attr = MaybeUninit::<libc::pthread_attr_t>::uninit();
        assert_eq!(libc::pthread_attr_init(attr.as_mut_ptr()), 0);

        // Too small. Round down to a page boundary so macOS cannot round it up to a valid size.
        let too_small = (libc::PTHREAD_STACK_MIN / 2) / page_size() * page_size();
        assert_eq!(libc::pthread_attr_setstacksize(attr.as_mut_ptr(), too_small), libc::EINVAL);
        // Not a multiple of the page size: only macOS rejects this.
        let ret = libc::pthread_attr_setstacksize(attr.as_mut_ptr(), normal_stack_size() + 1);
        if cfg!(target_os = "macos") {
            assert_eq!(ret, libc::EINVAL);
        } else {
            assert_eq!(ret, 0);
        }
        // A normal size.
        assert_eq!(libc::pthread_attr_setstacksize(attr.as_mut_ptr(), normal_stack_size()), 0);

        assert_eq!(libc::pthread_attr_destroy(attr.as_mut_ptr()), 0);
    }
}

fn test_create() {
    unsafe {
        // Default attributes.
        assert_eq!(spawn(ptr::null()), 0);

        let mut attr = MaybeUninit::<libc::pthread_attr_t>::uninit();
        assert_eq!(libc::pthread_attr_init(attr.as_mut_ptr()), 0);
        // Initialized attributes without a stack size.
        assert_eq!(spawn(attr.as_ptr()), 0);
        // A normal stack size.
        assert_eq!(libc::pthread_attr_setstacksize(attr.as_mut_ptr(), normal_stack_size()), 0);
        assert_eq!(spawn(attr.as_ptr()), 0);

        // A stack that is too big to reserve. Keep it page-aligned so macOS accepts it.
        let huge = if cfg!(target_pointer_width = "32") {
            usize::MAX - (2 * page_size() - 1)
        } else {
            (1 << (usize::BITS - 1)) + page_size()
        };
        assert_eq!(libc::pthread_attr_setstacksize(attr.as_mut_ptr(), huge), 0);
        let ret = spawn(attr.as_ptr());
        assert_eq!(ret, libc::EAGAIN);
        assert_eq!(libc::pthread_attr_destroy(attr.as_mut_ptr()), 0);
    }
}

fn main() {
    test_init_destroy();
    test_setstacksize();
    test_create();
}
