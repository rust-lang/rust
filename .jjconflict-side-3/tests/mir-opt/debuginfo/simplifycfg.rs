//@ test-mir-pass: SimplifyCfg-final
//@ compile-flags: -Zmir-enable-passes=+DeadStoreElimination-initial

#![feature(core_intrinsics, custom_mir)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

pub struct Foo {
    a: i32,
    b: i64,
    c: i32,
}

// EMIT_MIR simplifycfg.drop_debuginfo.SimplifyCfg-final.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn drop_debuginfo(foo: &Foo, c: bool) -> i32 {
    // CHECK-LABEL: fn drop_debuginfo
    // CHECK: debug foo_b => [[foo_b:_[0-9]+]];
    // CHECK: bb0: {
    // CHECK-NEXT: DBG: [[foo_b]] = &((*_1).1: i64)
    // CHECK-NEXT: _0 = copy ((*_1).2: i32);
    // CHECK-NEXT: return;
    mir! {
        let _foo_a: &i32;
        let _foo_b: &i64;
        debug foo_a => _foo_a;
        debug foo_b => _foo_b;
        {
            match c {
                true => tmp,
                _ => ret,
            }
        }
        tmp = {
            // Because we don't know if `c` is always true, we must drop this debuginfo.
            _foo_a = &(*foo).a;
            Goto(ret)
        }
        ret = {
            _foo_b = &(*foo).b;
            RET = (*foo).c;
            Return()
        }
    }
}

// EMIT_MIR simplifycfg.preserve_debuginfo_1.SimplifyCfg-final.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn preserve_debuginfo_1(foo: &Foo, v: &mut bool) -> i32 {
    // CHECK-LABEL: fn preserve_debuginfo_1
    // CHECK: debug foo_a => [[foo_a:_[0-9]+]];
    // CHECK: debug foo_b => [[foo_b:_[0-9]+]];
    // CHECK: debug foo_c => [[foo_c:_[0-9]+]];
    // CHECK: bb0: {
    // CHECK-NEXT: (*_2) = const true;
    // CHECK-NEXT: DBG: [[foo_a]] = &((*_1).0: i32)
    // CHECK-NEXT: DBG: [[foo_b]] = &((*_1).1: i64)
    // CHECK-NEXT: _0 = copy ((*_1).2: i32);
    // CHECK-NEXT: DBG: [[foo_c]] = &((*_1).2: i32)
    // CHECK-NEXT: return;
    mir! {
        let _foo_a: &i32;
        let _foo_b: &i64;
        let _foo_c: &i32;
        debug foo_a => _foo_a;
        debug foo_b => _foo_b;
        debug foo_c => _foo_c;
        {
            Goto(tmp)
        }
        tmp = {
            *v = true;
            _foo_a = &(*foo).a;
            Goto(ret)
        }
        ret = {
            _foo_b = &(*foo).b;
            RET = (*foo).c;
            _foo_c = &(*foo).c;
            Return()
        }
    }
}

// EMIT_MIR simplifycfg.preserve_debuginfo_2.SimplifyCfg-final.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn preserve_debuginfo_2(foo: &Foo) -> i32 {
    // CHECK-LABEL: fn preserve_debuginfo_2
    // CHECK: debug foo_a => [[foo_a:_[0-9]+]];
    // CHECK: debug foo_b => [[foo_b:_[0-9]+]];
    // CHECK: debug foo_c => [[foo_c:_[0-9]+]];
    // CHECK: bb0: {
    // CHECK-NEXT: DBG: [[foo_a]] = &((*_1).0: i32)
    // CHECK-NEXT: DBG: [[foo_b]] = &((*_1).1: i64)
    // CHECK-NEXT: _0 = copy ((*_1).2: i32);
    // CHECK-NEXT: DBG: [[foo_c]] = &((*_1).2: i32)
    // CHECK-NEXT: return;
    mir! {
        let _foo_a: &i32;
        let _foo_b: &i64;
        let _foo_c: &i32;
        debug foo_a => _foo_a;
        debug foo_b => _foo_b;
        debug foo_c => _foo_c;
        {
            Goto(tmp)
        }
        tmp = {
            _foo_a = &(*foo).a;
            Goto(ret)
        }
        ret = {
            _foo_b = &(*foo).b;
            RET = (*foo).c;
            _foo_c = &(*foo).c;
            Return()
        }
    }
}

// EMIT_MIR simplifycfg.preserve_debuginfo_3.SimplifyCfg-final.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn preserve_debuginfo_3(foo: &Foo, c: bool) -> i32 {
    // CHECK-LABEL: fn preserve_debuginfo_3
    // CHECK: debug foo_a => [[foo_a:_[0-9]+]];
    // CHECK: debug foo_b => [[foo_b:_[0-9]+]];
    // CHECK: debug foo_c => [[foo_c:_[0-9]+]];
    // CHECK: bb0: {
    // CHECK-NEXT: switchInt(copy _2) -> [1: bb2, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK-NEXT: DBG: [[foo_b]] = &((*_1).1: i64)
    // CHECK-NEXT: _0 = copy ((*_1).2: i32);
    // CHECK-NEXT: return;
    // CHECK: bb2: {
    // CHECK-NEXT: DBG: [[foo_a]] = &((*_1).0: i32)
    // CHECK-NEXT: DBG: [[foo_c]] = &((*_1).2: i32)
    // CHECK-NEXT: _0 = copy ((*_1).0: i32);
    // CHECK-NEXT: return;
    mir! {
        let _foo_a: &i32;
        let _foo_b: &i64;
        let _foo_c: &i32;
        debug foo_a => _foo_a;
        debug foo_b => _foo_b;
        debug foo_c => _foo_c;
        {
            match c {
                true => tmp,
                _ => ret,
            }
        }
        tmp = {
            _foo_a = &(*foo).a;
            Goto(ret_1)
        }
        ret = {
            _foo_b = &(*foo).b;
            RET = (*foo).c;
            Return()
        }
        ret_1 = {
            _foo_c = &(*foo).c;
            RET = (*foo).a;
            Return()
        }
    }
}

// EMIT_MIR simplifycfg.preserve_debuginfo_identical_succs.SimplifyCfg-final.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn preserve_debuginfo_identical_succs(foo: &Foo, c: bool) -> i32 {
    // CHECK-LABEL: fn preserve_debuginfo_identical_succs
    // CHECK: debug foo_a => [[foo_a:_[0-9]+]];
    // CHECK: debug foo_b => [[foo_b:_[0-9]+]];
    // CHECK: debug foo_c => [[foo_c:_[0-9]+]];
    // CHECK: bb0: {
    // CHECK-NEXT: DBG: [[foo_a]] = &((*_1).0: i32)
    // CHECK-NEXT: DBG: [[foo_b]] = &((*_1).1: i64)
    // CHECK-NEXT: _0 = copy ((*_1).2: i32);
    // CHECK-NEXT: DBG: [[foo_c]] = &((*_1).2: i32)
    // CHECK-NEXT: return;
    mir! {
        let _foo_a: &i32;
        let _foo_b: &i64;
        let _foo_c: &i32;
        debug foo_a => _foo_a;
        debug foo_b => _foo_b;
        debug foo_c => _foo_c;
        {
            match c {
                true => tmp,
                _ => tmp,
            }
        }
        tmp = {
            _foo_a = &(*foo).a;
            Goto(ret)
        }
        ret = {
            _foo_b = &(*foo).b;
            RET = (*foo).c;
            _foo_c = &(*foo).c;
            Return()
        }
    }
}
