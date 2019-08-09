// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

fn take_fn_once<T1, T2, F: FnOnce(T1, T2)>(f: F, x: T1, y: T2) {
    (f)(x, y)
}

fn function<T1, T2>(_: T1, _: T2) {}

fn take_fn_pointer<T1, T2>(f: fn(T1, T2), x: T1, y: T2) {
    (f)(x, y)
}

//~ MONO_ITEM fn function_as_argument::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {

    //~ MONO_ITEM fn function_as_argument::take_fn_once[0]<u32, &str, fn(u32, &str)>
    //~ MONO_ITEM fn function_as_argument::function[0]<u32, &str>
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(u32, &str), (u32, &str)>
    take_fn_once(function, 0u32, "abc");

    //~ MONO_ITEM fn function_as_argument::take_fn_once[0]<char, f64, fn(char, f64)>
    //~ MONO_ITEM fn function_as_argument::function[0]<char, f64>
    //~ MONO_ITEM fn core::ops[0]::function[0]::FnOnce[0]::call_once[0]<fn(char, f64), (char, f64)>
    take_fn_once(function, 'c', 0f64);

    //~ MONO_ITEM fn function_as_argument::take_fn_pointer[0]<i32, ()>
    //~ MONO_ITEM fn function_as_argument::function[0]<i32, ()>
    take_fn_pointer(function, 0i32, ());

    //~ MONO_ITEM fn function_as_argument::take_fn_pointer[0]<f32, i64>
    //~ MONO_ITEM fn function_as_argument::function[0]<f32, i64>
    take_fn_pointer(function, 0f32, 0i64);

    0
}
