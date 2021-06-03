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

// Make sure shims take precedence.
#[no_mangle]
extern "C" fn exit(_: i32) -> ! {
    unreachable!()
}

#[no_mangle]
extern "C" fn ExitProcess(_: u32) -> ! {
    unreachable!()
}

fn main() {
    // Repeat calls to make sure the `Instance` cache is not broken.
    for _ in 0..3 {
        extern "C" {
            fn foo() -> i32;
        }

        assert_eq!(unsafe { foo() }, -1);

        extern "Rust" {
            fn bar() -> i32;
            fn baz() -> i32;
        }

        assert_eq!(unsafe { bar() }, -2);
        assert_eq!(unsafe { baz() }, -3);

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
                fn baz() -> i32;
            }

            unsafe {
                let transmute = |f| {
                    std::mem::transmute::<unsafe extern "C" fn() -> i32, unsafe fn() -> i32>(f)
                };
                assert_eq!(transmute(bar)(), -2);
                assert_eq!(transmute(baz)(), -3);
            }
        }
    }
}
