//@ test-mir-pass: SingleUseConsts
//@ compile-flags: -C debuginfo=full
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

trait MyTrait {
    const ASSOC_BOOL: bool;
    const ASSOC_INT: i32;
}

// EMIT_MIR single_use_consts.if_const.SingleUseConsts.diff
fn if_const<T: MyTrait>() -> i32 {
    // CHECK-LABEL: fn if_const(
    // CHECK: switchInt(const <T as MyTrait>::ASSOC_BOOL)
    if T::ASSOC_BOOL { 7 } else { 42 }
}

// EMIT_MIR single_use_consts.match_const.SingleUseConsts.diff
fn match_const<T: MyTrait>() -> &'static str {
    // CHECK-LABEL: fn match_const(
    // CHECK: switchInt(const <T as MyTrait>::ASSOC_INT)
    match T::ASSOC_INT {
        7 => "hello",
        42 => "towel",
        _ => "world",
    }
}

// EMIT_MIR single_use_consts.if_const_debug.SingleUseConsts.diff
fn if_const_debug<T: MyTrait>() -> i32 {
    // CHECK-LABEL: fn if_const_debug(
    // CHECK: my_bool => const <T as MyTrait>::ASSOC_BOOL;
    // FIXME: `if` forces a temporary (unlike `match`), so the const isn't direct
    // CHECK: _3 = const <T as MyTrait>::ASSOC_BOOL;
    // CHECK: switchInt(move _3)
    let my_bool = T::ASSOC_BOOL;
    do_whatever();
    if my_bool { 7 } else { 42 }
}

// EMIT_MIR single_use_consts.match_const_debug.SingleUseConsts.diff
fn match_const_debug<T: MyTrait>() -> &'static str {
    // CHECK-LABEL: fn match_const_debug(
    // CHECK: my_int => const <T as MyTrait>::ASSOC_INT;
    // CHECK: switchInt(const <T as MyTrait>::ASSOC_INT)
    let my_int = T::ASSOC_INT;
    do_whatever();
    match my_int {
        7 => "hello",
        42 => "towel",
        _ => "world",
    }
}

// EMIT_MIR single_use_consts.never_used_debug.SingleUseConsts.diff
#[allow(unused_variables)]
fn never_used_debug<T: MyTrait>() {
    // CHECK-LABEL: fn never_used_debug(
    // CHECK: my_int => const <T as MyTrait>::ASSOC_INT;
    // CHECK-NOT: ASSOC_INT
    // CHECK: nop
    // CHECK-NOT: ASSOC_INT
    let my_int = T::ASSOC_INT;
}

// EMIT_MIR single_use_consts.assign_const_to_return.SingleUseConsts.diff
fn assign_const_to_return<T: MyTrait>() -> bool {
    // CHECK-LABEL: fn assign_const_to_return(
    // CHECK: _0 = const <T as MyTrait>::ASSOC_BOOL;
    T::ASSOC_BOOL
}

// EMIT_MIR single_use_consts.keep_parameter.SingleUseConsts.diff
fn keep_parameter<T: MyTrait>(mut other: i32) {
    // CHECK-LABEL: fn keep_parameter(
    // CHECK: _1 = const <T as MyTrait>::ASSOC_INT;
    // CHECK: _0 = const ();
    other = T::ASSOC_INT;
}

fn do_whatever() {}
