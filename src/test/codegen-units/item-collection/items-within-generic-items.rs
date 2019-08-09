// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

fn generic_fn<T>(a: T) -> (T, i32) {
    //~ MONO_ITEM fn items_within_generic_items::generic_fn[0]::nested_fn[0]
    fn nested_fn(a: i32) -> i32 {
        a + 1
    }

    let x = {
        //~ MONO_ITEM fn items_within_generic_items::generic_fn[0]::nested_fn[1]
        fn nested_fn(a: i32) -> i32 {
            a + 2
        }

        1 + nested_fn(1)
    };

    return (a, x + nested_fn(0));
}

//~ MONO_ITEM fn items_within_generic_items::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn items_within_generic_items::generic_fn[0]<i64>
    let _ = generic_fn(0i64);
    //~ MONO_ITEM fn items_within_generic_items::generic_fn[0]<u16>
    let _ = generic_fn(0u16);
    //~ MONO_ITEM fn items_within_generic_items::generic_fn[0]<i8>
    let _ = generic_fn(0i8);

    0
}
