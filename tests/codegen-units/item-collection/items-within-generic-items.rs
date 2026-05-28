//@ compile-flags:-Clink-dead-code -Copt-level=0

#![deny(dead_code)]
#![crate_type = "lib"]

fn generic_fn<T>(a: T) -> (T, i32) {
    //~ MONO_ITEM fn generic_fn::nested_fn
    fn nested_fn(a: i32) -> i32 {
        a + 1
    }

    let x = {
        //~ MONO_ITEM fn generic_fn::nested_fn
        fn nested_fn(a: i32) -> i32 {
            a + 2
        }

        1 + nested_fn(1)
    };

    return (a, x + nested_fn(0));
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn generic_fn::<i64>
    let _ = generic_fn(0i64);
    //~ MONO_ITEM fn generic_fn::<u16>
    let _ = generic_fn(0u16);
    //~ MONO_ITEM fn generic_fn::<i8>
    let _ = generic_fn(0i8);

    0
}
