//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

fn take_fn_once<T1, T2, F: FnOnce(T1, T2)>(f: F, x: T1, y: T2) {
    (f)(x, y)
}

fn function<T1, T2>(_: T1, _: T2) {}

fn take_fn_pointer<T1, T2>(f: fn(T1, T2), x: T1, y: T2) {
    (f)(x, y)
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn take_fn_once::<u32, &str, fn(u32, &str) {function::<u32, &str>}>
    //~ MONO_ITEM fn function::<u32, &str>
    //~ MONO_ITEM fn <fn(u32, &str) {function::<u32, &str>} as std::ops::FnOnce<(u32, &str)>>::call_once - shim(fn(u32, &str) {function::<u32, &str>})
    take_fn_once(function, 0u32, "abc");

    //~ MONO_ITEM fn take_fn_once::<char, f64, fn(char, f64) {function::<char, f64>}>
    //~ MONO_ITEM fn function::<char, f64>
    //~ MONO_ITEM fn <fn(char, f64) {function::<char, f64>} as std::ops::FnOnce<(char, f64)>>::call_once - shim(fn(char, f64) {function::<char, f64>})
    take_fn_once(function, 'c', 0f64);

    //~ MONO_ITEM fn take_fn_pointer::<i32, ()>
    //~ MONO_ITEM fn function::<i32, ()>
    take_fn_pointer(function, 0i32, ());

    //~ MONO_ITEM fn take_fn_pointer::<f32, i64>
    //~ MONO_ITEM fn function::<f32, i64>
    take_fn_pointer(function, 0f32, 0i64);

    0
}
