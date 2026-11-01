#![feature(rustc_attrs)]

#[no_mangle]
extern "C" fn foo() -> i32 {
    -1
}

#[export_name = "bar"]
fn bar() -> i32 {
    -2
}

#[rustc_std_internal_symbol]
fn baz() -> i32 {
    -3
}

struct AssocFn;

impl AssocFn {
    #[no_mangle]
    fn qux() -> i32 {
        -4
    }
}

mod somewhere_else {
    #[repr(C)]
    struct CType {
        field: i32,
    }

    #[unsafe(no_mangle)]
    extern "C" fn work_with_c_type(_x: CType) {}
}

// Imagine we import the function from above but we end up with a copy of the type declaration.
#[repr(C)]
struct CType {
    field: i32,
}
extern "C" {
    fn work_with_c_type(_x: CType);
}

fn main() {
    // Repeat calls to make sure the `Instance` cache is not broken.
    for _ in 0..3 {
        extern "C" {
            fn foo() -> i32;
            fn free(_: *mut std::ffi::c_void);
        }

        assert_eq!(unsafe { foo() }, -1);

        // `free()` is a built-in shim, so calling it will add ("free", None) to the cache.
        // Test that the cache is not broken with ("free", None).
        unsafe { free(std::ptr::null_mut()) }

        extern "Rust" {
            fn bar() -> i32;
            #[rustc_std_internal_symbol]
            fn baz() -> i32;
            fn qux() -> i32;
        }

        assert_eq!(unsafe { bar() }, -2);
        assert_eq!(unsafe { baz() }, -3);
        assert_eq!(unsafe { qux() }, -4);

        #[allow(clashing_extern_declarations)]
        {
            extern "Rust" {
                fn foo() -> i32;
            }

            assert_eq!(
                unsafe {
                    std::mem::transmute::<unsafe fn() -> i32, unsafe extern "C" fn() -> i32>(foo)()
                },
                -1
            );

            extern "C" {
                fn bar() -> i32;
                #[rustc_std_internal_symbol]
                fn baz() -> i32;
                fn qux() -> i32;
            }

            unsafe {
                let transmute =
                    |f| std::mem::transmute::<unsafe extern "C" fn() -> i32, unsafe fn() -> i32>(f);
                assert_eq!(transmute(bar)(), -2);
                assert_eq!(transmute(baz)(), -3);
                assert_eq!(transmute(qux)(), -4);
            }
        }

        unsafe { work_with_c_type(CType { field: 0 }) };
    }
}
