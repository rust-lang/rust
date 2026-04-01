//@ compile-flags:-Clink-dead-code -Zinline-mir=no -O

#![deny(dead_code)]
#![crate_type = "lib"]

//~ MONO_ITEM fn temporary @@ non_generic_closures-cgu.0[External]
fn temporary() {
    //~ MONO_ITEM fn temporary::{closure#0} @@ non_generic_closures-cgu.0[External]
    (|a: u32| {
        let _ = a;
    })(4);
}

//~ MONO_ITEM fn assigned_to_variable_but_not_executed @@ non_generic_closures-cgu.0[External]
fn assigned_to_variable_but_not_executed() {
    //~ MONO_ITEM fn assigned_to_variable_but_not_executed::{closure#0}
    let _x = |a: i16| {
        let _ = a + 1;
    };
}

//~ MONO_ITEM fn assigned_to_variable_executed_indirectly @@ non_generic_closures-cgu.0[External]
fn assigned_to_variable_executed_indirectly() {
    //~ MONO_ITEM fn assigned_to_variable_executed_indirectly::{closure#0} @@ non_generic_closures-cgu.0[External]
    //~ MONO_ITEM fn <{closure@TEST_PATH:27:13: 27:21} as std::ops::FnOnce<(i32,)>>::call_once - shim @@ non_generic_closures-cgu.0[External]
    //~ MONO_ITEM fn <{closure@TEST_PATH:27:13: 27:21} as std::ops::FnOnce<(i32,)>>::call_once - shim(vtable) @@ non_generic_closures-cgu.0[External]
    let f = |a: i32| {
        let _ = a + 2;
    };
    run_closure(&f);
}

//~ MONO_ITEM fn assigned_to_variable_executed_directly @@ non_generic_closures-cgu.0[External]
fn assigned_to_variable_executed_directly() {
    //~ MONO_ITEM fn assigned_to_variable_executed_directly::{closure#0} @@ non_generic_closures-cgu.0[External]
    let f = |a: i64| {
        let _ = a + 3;
    };
    f(4);
}

// Make sure we generate mono items for stateful closures that need dropping
//~ MONO_ITEM fn with_drop @@ non_generic_closures-cgu.0[External]
fn with_drop(v: PresentDrop) {
    //~ MONO_ITEM fn with_drop::{closure#0} @@ non_generic_closures-cgu.0[External]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<PresentDrop> - shim(Some(PresentDrop)) @@ non_generic_closures-cgu.0[Internal]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<{closure@TEST_PATH:49:14: 49:24}> - shim(Some({closure@TEST_PATH:49:14: 49:24})) @@ non_generic_closures-cgu.0[Internal]

    let _f = |a: usize| {
        let _ = a + 2;
        //~ MONO_ITEM fn std::mem::drop::<PresentDrop> @@ non_generic_closures-cgu.0[External]
        drop(v);
    };
}

//~ MONO_ITEM fn start @@ non_generic_closures-cgu.0[External]
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    temporary();
    assigned_to_variable_but_not_executed();
    assigned_to_variable_executed_directly();
    assigned_to_variable_executed_indirectly();
    with_drop(PresentDrop);

    0
}

//~ MONO_ITEM fn run_closure @@ non_generic_closures-cgu.0[External]
fn run_closure(f: &Fn(i32)) {
    f(3);
}

struct PresentDrop;

impl Drop for PresentDrop {
    //~ MONO_ITEM fn <PresentDrop as std::ops::Drop>::drop @@ non_generic_closures-cgu.0[External]
    fn drop(&mut self) {}
}
