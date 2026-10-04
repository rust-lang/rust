//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

fn foo1<T1>(a: T1) -> (T1, u32) {
    (a, 1)
}

fn foo2<T1, T2>(a: T1, b: T2) -> (T1, T2) {
    (a, b)
}

fn foo3<T1, T2, T3>(a: T1, b: T2, c: T3) -> (T1, T2, T3) {
    (a, b, c)
}

// This function should be instantiated even if no used
//~ MONO_ITEM fn lifetime_only
pub fn lifetime_only<'a>(a: &'a u32) -> &'a u32 {
    a
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn foo1::<i32>
    let _ = foo1(2i32);
    //~ MONO_ITEM fn foo1::<i64>
    let _ = foo1(2i64);
    //~ MONO_ITEM fn foo1::<&str>
    let _ = foo1("abc");
    //~ MONO_ITEM fn foo1::<char>
    let _ = foo1('v');

    //~ MONO_ITEM fn foo2::<i32, i32>
    let _ = foo2(2i32, 2i32);
    //~ MONO_ITEM fn foo2::<i64, &str>
    let _ = foo2(2i64, "abc");
    //~ MONO_ITEM fn foo2::<&str, usize>
    let _ = foo2("a", 2usize);
    //~ MONO_ITEM fn foo2::<char, ()>
    let _ = foo2('v', ());

    //~ MONO_ITEM fn foo3::<i32, i32, i32>
    let _ = foo3(2i32, 2i32, 2i32);
    //~ MONO_ITEM fn foo3::<i64, &str, char>
    let _ = foo3(2i64, "abc", 'c');
    //~ MONO_ITEM fn foo3::<i16, &str, usize>
    let _ = foo3(0i16, "a", 2usize);
    //~ MONO_ITEM fn foo3::<char, (), ()>
    let _ = foo3('v', (), ());

    0
}
