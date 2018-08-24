// In the current version of the collector that still has to support
// legacy-codegen, closures do not generate their own MonoItems, so we are
// ignoring this test until MIR codegen has taken over completely
// ignore-test

// ignore-tidy-linelength
// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

//~ MONO_ITEM fn non_generic_closures::temporary[0]
fn temporary() {
    //~ MONO_ITEM fn non_generic_closures::temporary[0]::{{closure}}[0]
    (|a: u32| {
        let _ = a;
    })(4);
}

//~ MONO_ITEM fn non_generic_closures::assigned_to_variable_but_not_executed[0]
fn assigned_to_variable_but_not_executed() {
    //~ MONO_ITEM fn non_generic_closures::assigned_to_variable_but_not_executed[0]::{{closure}}[0]
    let _x = |a: i16| {
        let _ = a + 1;
    };
}

//~ MONO_ITEM fn non_generic_closures::assigned_to_variable_executed_directly[0]
fn assigned_to_variable_executed_indirectly() {
    //~ MONO_ITEM fn non_generic_closures::assigned_to_variable_executed_directly[0]::{{closure}}[0]
    let f = |a: i32| {
        let _ = a + 2;
    };
    run_closure(&f);
}

//~ MONO_ITEM fn non_generic_closures::assigned_to_variable_executed_indirectly[0]
fn assigned_to_variable_executed_directly() {
    //~ MONO_ITEM fn non_generic_closures::assigned_to_variable_executed_indirectly[0]::{{closure}}[0]
    let f = |a: i64| {
        let _ = a + 3;
    };
    f(4);
}

//~ MONO_ITEM fn non_generic_closures::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    temporary();
    assigned_to_variable_but_not_executed();
    assigned_to_variable_executed_directly();
    assigned_to_variable_executed_indirectly();

    0
}

//~ MONO_ITEM fn non_generic_closures::run_closure[0]
fn run_closure(f: &Fn(i32)) {
    f(3);
}
