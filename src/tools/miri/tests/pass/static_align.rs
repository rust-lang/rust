#![feature(static_align)]
#![deny(non_upper_case_globals)]

use std::cell::Cell;

// When a static uses `align(N)`, its address should be a multiple of `N`.

#[rustc_align_static(256)]
static FOO: u64 = 0;

#[rustc_align_static(512)]
static BAR: u64 = 0;

struct HasDrop(*const HasDrop);

impl Drop for HasDrop {
    fn drop(&mut self) {
        assert_eq!(core::ptr::from_mut(self).cast_const(), self.0);
    }
}

thread_local! {
    #[rustc_align_static(4096)]
    static LOCAL: u64 = 0;

    #[allow(unused_mut, reason = "test attribute handling")]
    #[cfg_attr(true, rustc_align_static(4096))]
    static CONST_LOCAL: u64 = const { 0 };

    #[cfg_attr(any(true), cfg_attr(true, rustc_align_static(4096)))]
    #[allow(unused_mut, reason = "test attribute handling")]
    static HASDROP_LOCAL: Cell<HasDrop> = Cell::new(HasDrop(core::ptr::null()));

    /// I love doc comments.
    #[allow(unused_mut, reason = "test attribute handling")]
    #[cfg_attr(all(),
      cfg_attr(any(true),
      cfg_attr(true, rustc_align_static(4096))))]
    #[allow(unused_mut, reason = "test attribute handling")]
    /// I love doc comments.
    static HASDROP_CONST_LOCAL: Cell<HasDrop> = const { Cell::new(HasDrop(core::ptr::null())) };

    #[cfg_attr(true,)]
    #[cfg_attr(false,)]
    #[cfg_attr(
        true,
        rustc_align_static(32),
        cfg_attr(true, allow(non_upper_case_globals, reason = "test attribute handling")),
        cfg_attr(false,)
    )]
    #[cfg_attr(false, rustc_align_static(0))]
    static more_attr_testing: u64 = 0;
}

fn thread_local_ptr<T>(key: &'static std::thread::LocalKey<T>) -> *const T {
    key.with(|local| core::ptr::from_ref::<T>(local))
}

fn main() {
    assert!(core::ptr::from_ref(&FOO).addr().is_multiple_of(256));
    assert!(core::ptr::from_ref(&BAR).addr().is_multiple_of(512));

    assert!(thread_local_ptr(&LOCAL).addr().is_multiple_of(4096));
    assert!(thread_local_ptr(&CONST_LOCAL).addr().is_multiple_of(4096));
    assert!(thread_local_ptr(&HASDROP_LOCAL).addr().is_multiple_of(4096));
    assert!(thread_local_ptr(&HASDROP_CONST_LOCAL).addr().is_multiple_of(4096));
    assert!(thread_local_ptr(&more_attr_testing).addr().is_multiple_of(32));

    // Test that address (and therefore alignment) is maintained during drop
    let hasdrop_ptr = thread_local_ptr(&HASDROP_LOCAL);
    core::mem::forget(HASDROP_LOCAL.replace(HasDrop(hasdrop_ptr.cast())));
    let hasdrop_const_ptr = thread_local_ptr(&HASDROP_CONST_LOCAL);
    core::mem::forget(HASDROP_CONST_LOCAL.replace(HasDrop(hasdrop_const_ptr.cast())));
}
