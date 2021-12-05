// revisions: CHECK-BASE CHECK-OPT
// compile-flags: -C no-prepopulate-passes -C opt-level=0 -Z mir-opt-level=1
//[CHECK-BASE] compile-flags: -Z opt-switch-monomorphizing=off
//[CHECK-OPT] compile-flags: -Z opt-switch-monomorphizing=on

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(never_type)]

use std::intrinsics::discriminant_value;
use std::num::NonZeroUsize;

pub enum BothEmpty {
    Left(!),
    Right(!),
}

// CHECK-LABEL: @match_both_empty
#[no_mangle]
pub fn match_both_empty(e: BothEmpty) -> u8 {
    // CHECK: %e =
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] undef, label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT:    i[[SIZE]] 0
    // CHECK-BASE-NEXT:    i[[SIZE]] 1
    // CHECK-BASE-NEXT:    i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: unreachable
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum EmptyOrBool {
    Empty(!),
    Bool(bool),
}

// CHECK-LABEL: @match_empty_or_bool
#[no_mangle]
pub fn match_empty_or_bool(e: EmptyOrBool) -> u8 {
    // CHECK: store i8 %[[D:[0-9]+]], i8* %e
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] 1, label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0
    // CHECK-BASE-NEXT: i[[SIZE]] 1, label %[[R:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: br label %[[R:[a-zA-Z0-9_]+]]
    // CHECK: [[R]]:
    // CHECK-NEXT: store i8 101, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum BoolOrEmpty {
    Bool(bool),
    Empty(!),
}

// CHECK-LABEL: @match_bool_or_empty
#[no_mangle]
pub fn match_bool_or_empty(e: BoolOrEmpty) -> u8 {
    // CHECK: store i8 %[[D:[0-9]+]], i8* %e
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] 0, label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0, label %[[L:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 1
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: br label %[[L:[a-zA-Z0-9_]+]]
    // CHECK: [[L]]:
    // CHECK-NEXT: store i8 100, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum UninhabitedUsizeOrBool {
    Usize(usize, !),
    Bool(bool),
}

// CHECK-LABEL: @match_uninhabited_usize_or_bool
#[no_mangle]
pub fn match_uninhabited_usize_or_bool(e: UninhabitedUsizeOrBool) -> u8 {
    // CHECK: %[[TAG:[0-9]+]] = load i8, i8* %{{[0-9]+}}
    // CHECK-NEXT: %_[[D:[0-9]+]] = zext i8 %[[TAG]] to i[[SIZE:[0-9]+]]
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] %_[[D]], label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0
    // CHECK-BASE-NEXT: i[[SIZE]] 1, label %[[R:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: %[[CMP:[0-9]+]] = icmp eq i[[SIZE]] %_[[D]], 1
    // CHECK-OPT-NEXT: br i1 %[[CMP]], label %[[R:[a-zA-Z0-9_]+]]
    // CHECK: [[R]]:
    // CHECK-NEXT: store i8 101, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum BoolOrUninhabitedUsize {
    Bool(bool),
    Usize(usize, !),
}

// CHECK-LABEL: @match_bool_or_uninhabited_nonzero_usize
#[no_mangle]
pub fn match_bool_or_uninhabited_nonzero_usize(e: BoolOrUninhabitedUsize) -> u8 {
    // CHECK: %[[TAG:[0-9]+]] = load i8, i8* %{{[0-9]+}}
    // CHECK-NEXT: %_[[D:[0-9]+]] = zext i8 %[[TAG]] to i[[SIZE:[0-9]+]]
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] %_[[D]], label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0, label %[[L:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 1
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: %[[CMP:[0-9]+]] = icmp eq i[[SIZE]] %_[[D]], 0
    // CHECK-OPT-NEXT: br i1 %[[CMP]], label %[[L:[a-zA-Z0-9_]+]]
    // CHECK: [[L]]:
    // CHECK-NEXT: store i8 100, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum UninhabitedNonZeroUsizeOrUnit {
    Usize(NonZeroUsize, !),
    Unit,
}

// CHECK-LABEL: @match_uninhabited_non_zero_usize_or_unit
#[no_mangle]
pub fn match_uninhabited_non_zero_usize_or_unit(e: UninhabitedNonZeroUsizeOrUnit) -> u8 {
    // CHECK: %[[TAG:[0-9]+]] = load i[[SIZE:[0-9]+]], i[[SIZE]]* %{{[0-9]+}}
    // CHECK-NEXT: %[[TMP:[0-9]+]] = icmp eq i[[SIZE]] %[[TAG]], 0
    // CHECK-NEXT: %_[[D:[0-9]+]] = select i1 %[[TMP]], i[[SIZE]] 1, i[[SIZE]] 0
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] %_[[D]], label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0
    // CHECK-BASE-NEXT: i[[SIZE]] 1, label %[[R:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: %[[CMP:[0-9]+]] = icmp eq i[[SIZE]] %_[[D]], 1
    // CHECK-OPT-NEXT: br i1 %[[CMP]], label %[[R:[a-zA-Z0-9_]+]]
    // CHECK: [[R]]:
    // CHECK-NEXT: store i8 101, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}

pub enum UnitOrUninhabitedNonZeroUsize {
    Unit,
    Usize(NonZeroUsize, !),
}

// CHECK-LABEL: @match_unit_or_uninhabited_non_zero_usize
#[no_mangle]
pub fn match_unit_or_uninhabited_non_zero_usize(e: UnitOrUninhabitedNonZeroUsize) -> u8 {
    // CHECK: %[[TAG:[0-9]+]] = load i[[SIZE:[0-9]+]], i[[SIZE]]* %{{[0-9]+}}
    // CHECK-NEXT: %[[TMP:[0-9]+]] = icmp eq i[[SIZE]] %[[TAG]], 0
    // CHECK-NEXT: %_[[D:[0-9]+]] = select i1 %[[TMP]], i[[SIZE]] 0, i[[SIZE]] 1
    // CHECK-BASE-NEXT: switch i[[SIZE:[0-9]+]] %_[[D]], label %{{[a-zA-Z0-9_]+}} [
    // CHECK-BASE-NEXT: i[[SIZE]] 0, label %[[R:[a-zA-Z0-9_]+]]
    // CHECK-BASE-NEXT: i[[SIZE]] 1
    // CHECK-BASE-NEXT: i[[SIZE]] 2
    // CHECK-BASE-NEXT: ]
    // CHECK-OPT-NEXT: %[[CMP:[0-9]+]] = icmp eq i[[SIZE]] %_[[D]], 0
    // CHECK-OPT-NEXT: br i1 %[[CMP]], label %[[R:[a-zA-Z0-9_]+]]
    // CHECK: [[R]]:
    // CHECK-NEXT: store i8 100, i8* %1
    match discriminant_value(&e) {
        0 => 100,
        1 => 101,
        2 => 102,
        _ => 103,
    }
}
