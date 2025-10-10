//@revisions: trace notrace
//@[trace] only-target: x86_64-unknown-linux-gnu i686-unknown-linux-gnu
//@[trace] compile-flags: -Zmiri-native-lib-enable-tracing
//@compile-flags: -Zmiri-permissive-provenance

fn main() {
    test_access_pointer();
    test_access_simple();
    test_access_nested();
    test_access_static();
}

/// Test function that dereferences an int pointer and prints its contents from C.
fn test_access_pointer() {
    extern "C" {
        fn print_pointer(ptr: *const i32);
    }

    let x = 42;

    unsafe { print_pointer(&x) };
}

/// Test function that dereferences a simple struct pointer and accesses a field.
fn test_access_simple() {
    #[repr(C)]
    struct Simple {
        field: i32,
    }

    extern "C" {
        fn access_simple(s_ptr: *const Simple) -> i32;
    }

    let simple = Simple { field: -42 };

    assert_eq!(unsafe { access_simple(&simple) }, -42);
}

/// Test function that dereferences nested struct pointers and accesses fields.
fn test_access_nested() {
    use std::ptr::NonNull;

    #[derive(Debug, PartialEq, Eq)]
    #[repr(C)]
    struct Nested {
        value: i32,
        next: Option<NonNull<Nested>>,
    }

    extern "C" {
        fn access_nested(n_ptr: *const Nested) -> i32;
    }

    let mut nested_0 = Nested { value: 97, next: None };
    let mut nested_1 = Nested { value: 98, next: NonNull::new(&mut nested_0) };
    let nested_2 = Nested { value: 99, next: NonNull::new(&mut nested_1) };

    assert_eq!(unsafe { access_nested(&nested_2) }, 97);
}

/// Test function that dereferences a static struct pointer and accesses fields.
fn test_access_static() {
    #[repr(C)]
    struct Static {
        value: i32,
        recurse: &'static Static,
    }

    extern "C" {
        fn access_static(n_ptr: *const Static) -> i32;
    }

    static STATIC: Static = Static { value: 9001, recurse: &STATIC };

    assert_eq!(unsafe { access_static(&STATIC) }, 9001);
}
