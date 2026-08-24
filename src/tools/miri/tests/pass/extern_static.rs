#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;

#[no_mangle]
static FOO: u8 = 42;

#[export_name = "BAR_EXPORTED"]
static BAR_LOCAL_NAME: u16 = 1000;

#[no_mangle]
static mut MUTABLE_STATIC: i32 = -1;

#[export_name = "MY_LINK_NAME"]
static RUST_SYMBOL: u32 = 7;

#[no_mangle]
static FOO_U32: u32 = 42;

#[no_mangle]
static INTERIOR_MUT: SyncUnsafeCell<i32> = SyncUnsafeCell::new(42);

fn increase_mutable_static_by_original_def(add_val: i32) {
    unsafe {
        let new_val = (&raw mut MUTABLE_STATIC).read() + add_val;
        (&raw mut MUTABLE_STATIC).write(new_val);
    }
}

fn main() {
    // The loop ensures we hit both the uncached and cached case.
    for _ in 0..3 {
        extern "Rust" {
            static FOO: u8;
        }

        assert_eq!(unsafe { (&raw const FOO).read() }, 42);

        extern "C" {
            static BAR_EXPORTED: u16;
        }

        assert_eq!(unsafe { (&raw const BAR_EXPORTED).read() }, 1000);

        extern "C" {
            #[link_name = "MY_LINK_NAME"]
            static EXTERN_STATIC: u32;
        }

        assert_eq!(unsafe { (&raw const EXTERN_STATIC).read() }, 7);

        // Ensure that SyncUnsafeCell and `static mut` are interchangable.
        extern "C" {
            #[link_name = "INTERIOR_MUT"]
            static mut INTERIOR_MUT_AS_MUTABLE_STATIC: i32;
            #[link_name = "MUTABLE_STATIC"]
            static MUTABLE_STATIC_AS_INTERIOR_MUT: SyncUnsafeCell<i32>;
        }
        unsafe {
            (&raw mut INTERIOR_MUT_AS_MUTABLE_STATIC).write(7);
            MUTABLE_STATIC_AS_INTERIOR_MUT.get().write(3);
        }
    }

    extern "Rust" {
        static mut MUTABLE_STATIC: i32;
    }

    // Check what happens if we mix accesses via the two aliases: the original
    // definition at the top of the file, and the extern declaration just above.
    unsafe {
        assert_eq!((&raw const MUTABLE_STATIC).read(), 3);
        (&raw mut MUTABLE_STATIC).write(32);
        increase_mutable_static_by_original_def(10);
        assert_eq!((&raw const MUTABLE_STATIC).read(), 42);
    }

    extern "Rust" {
        static FOO_U32: i32;
    }
    // This is like a transmute between raw pointers, so not UB.
    assert_eq!(unsafe { (&raw const FOO_U32).read() }, 42i32);
}
