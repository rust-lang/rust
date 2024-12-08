//@ compile-flags:-Zprint-mono-items=eager -Zinline-mir=no

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn temporary @@ non_generic_closures-cgu.0[Internal]
fn temporary() {
    //~ MONO_ITEM fn temporary::{closure#0} @@ non_generic_closures-cgu.0[Internal]
    (|a: u32| {
        let _ = a;
    })(4);
}

//~ MONO_ITEM fn assigned_to_variable_but_not_executed @@ non_generic_closures-cgu.0[Internal]
fn assigned_to_variable_but_not_executed() {
    let _x = |a: i16| {
        let _ = a + 1;
    };
}

//~ MONO_ITEM fn assigned_to_variable_executed_indirectly @@ non_generic_closures-cgu.0[Internal]
fn assigned_to_variable_executed_indirectly() {
    //~ MONO_ITEM fn assigned_to_variable_executed_indirectly::{closure#0} @@ non_generic_closures-cgu.0[Internal]
    //~ MONO_ITEM fn <{closure@TEST_PATH:27:13: 27:21} as std::ops::FnOnce<(i32,)>>::call_once - shim @@ non_generic_closures-cgu.0[Internal]
    //~ MONO_ITEM fn <{closure@TEST_PATH:27:13: 27:21} as std::ops::FnOnce<(i32,)>>::call_once - shim(vtable) @@ non_generic_closures-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<{closure@TEST_PATH:27:13: 27:21}> - shim(None) @@ non_generic_closures-cgu.0[Internal]
    let f = |a: i32| {
        let _ = a + 2;
    };
    run_closure(&f);
}

//~ MONO_ITEM fn assigned_to_variable_executed_directly @@ non_generic_closures-cgu.0[Internal]
fn assigned_to_variable_executed_directly() {
    //~ MONO_ITEM fn assigned_to_variable_executed_directly::{closure#0} @@ non_generic_closures-cgu.0[Internal]
    let f = |a: i64| {
        let _ = a + 3;
    };
    f(4);
}

//~ MONO_ITEM fn start @@ non_generic_closures-cgu.0[Internal]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    temporary();
    assigned_to_variable_but_not_executed();
    assigned_to_variable_executed_directly();
    assigned_to_variable_executed_indirectly();

    0
}

//~ MONO_ITEM fn run_closure @@ non_generic_closures-cgu.0[Internal]
fn run_closure(f: &Fn(i32)) {
    f(3);
}
