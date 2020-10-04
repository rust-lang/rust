// run-pass
// edition:2018

#![feature(once_cell)]
#![deny(const_item_mutation)]

mod library {
    use std::lazy::SyncLazy;
    use std::ops::{Deref, DerefMut};
    use std::ptr;
    use std::sync::atomic::{AtomicPtr, Ordering};
    use std::sync::Mutex;

    pub struct Options {
        pub include_prefix: &'static str,
    }

    static INCLUDE_PREFIX: SyncLazy<Mutex<&'static str>> = SyncLazy::new(Mutex::default);

    impl Options {
        fn current() -> Self {
            Options {
                include_prefix: *INCLUDE_PREFIX.lock().unwrap(),
            }
        }
    }

    pub struct Cfg {
        options: AtomicPtr<Options>,
    }

    pub const CFG: Cfg = Cfg {
        options: AtomicPtr::new(ptr::null_mut()),
    };

    impl Deref for Cfg {
        type Target = Options;
        fn deref(&self) -> &Self::Target {
            let options = Box::into_raw(Box::new(Options::current()));
            let prev = self
                .options
                .compare_and_swap(ptr::null_mut(), options, Ordering::Relaxed);
            if !prev.is_null() {
                // compare_and_swap did nothing.
                let _ = unsafe { Box::from_raw(options) };
                panic!();
            }
            unsafe { &*options }
        }
    }

    impl DerefMut for Cfg {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let options = self.options.get_mut();
            if !options.is_null() {
                panic!();
            }
            *options = Box::into_raw(Box::new(Options::current()));
            unsafe { &mut **options }
        }
    }

    impl Drop for Cfg {
        fn drop(&mut self) {
            let options = *self.options.get_mut();
            if let Some(options) = unsafe { options.as_mut() } {
                *INCLUDE_PREFIX.lock().unwrap() = options.include_prefix;
                let _ = unsafe { Box::from_raw(options) };
            }
        }
    }
}

use library::CFG;

fn main() {
    assert_eq!(CFG.include_prefix, "");

    CFG.include_prefix = "path/to";

    assert_eq!(CFG.include_prefix, "path/to");
}
