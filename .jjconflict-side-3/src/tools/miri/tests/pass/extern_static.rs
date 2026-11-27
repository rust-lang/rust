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

#[no_mangle]
static ARRAY: [u32; 5] = [1, 2, 3, 4, 5];

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

        // It's okay for the actual static to be bigger or more aligned than the extern declaration.
        extern "C" {
            // Actual size is bigger (20 bytes).
            #[link_name = "ARRAY"]
            static ARRAY_UNKNOWN_SIZE: [u32; 0];

            // Actual size and alignment is that of u32, not u16.
            #[link_name = "FOO_U32"]
            static U16_TO_FOO_U32: u16;
        }

        unsafe {
            let ptr = (&raw const ARRAY_UNKNOWN_SIZE).cast::<i32>();
            assert_eq!(ptr.read(), 1);
            assert_eq!(ptr.offset(2).read(), 3);

            // We see one half of FOO_U32, depending on endianess.
            if cfg!(target_endian = "little") {
                assert_eq!(U16_TO_FOO_U32, 42);
            } else {
                assert_eq!(U16_TO_FOO_U32, 0);
            }
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
