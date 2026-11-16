//@ needs-unwind

#![feature(core_intrinsics, custom_mir, rustc_attrs)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// A non-borrowed local dies on its last use.
// EMIT_MIR precise_liveness.unborrowed.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn unborrowed() -> u32 {
    // CHECK-LABEL: fn unborrowed(
    // CHECK: [[A:_[0-9]+]] = const 1_u32;
    // CHECK: [[B:_[0-9]+]] = const 2_u32;
    // CHECK: // early: [[[A]], [[B]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = Add(copy [[A]], move [[B]]);
    mir! {
        let a: u32;
        let b: u32;
        {
            a = 1;
            b = 2;
            RET = a + Move(b);
            Return()
        }
    }
}

// A borrowed local only dies on move.
// EMIT_MIR precise_liveness.borrowed.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn borrowed() -> u32 {
    // CHECK-LABEL: fn borrowed(
    // CHECK: [[A:_[0-9]+]] = const 1_u32;
    // CHECK: [[B:_[0-9]+]] = const 2_u32;
    // CHECK: = &raw const [[B]];
    // CHECK-NEXT: // early: [[[A]], [[B]]]
    // CHECK-NEXT: // late: [_0, [[A]]]
    // CHECK-NEXT: _0 = Add(copy [[A]], move [[B]]);
    mir! {
        let a: u32;
        let b: u32;
        let a_ptr: *const u32;
        let b_ptr: *const u32;
        {
            a = 1;
            b = 2;
            a_ptr = &raw const a;
            b_ptr = &raw const b;
            RET = a + Move(b);
            Return()
        }
    }
}

// StorageDead kills a lifetime, but it can be re-initialized later.
// EMIT_MIR precise_liveness.storage.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn storage() -> u32 {
    // CHECK-LABEL: fn storage(
    // CHECK: // early: [_0, [[BORROWED:_[0-9]+]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: StorageDead([[BORROWED]]);
    // CHECK-NEXT: // early: [_0]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: StorageLive([[BORROWED]]);
    // CHECK-NEXT: // early: [_0]
    // CHECK-NEXT: // late: [_0, [[BORROWED]]]
    // CHECK-NEXT: [[BORROWED]] = const 3_u32;
    // CHECK-NEXT: // early: [_0, [[BORROWED]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: StorageDead([[BORROWED]]);
    // CHECK-NEXT: // early: [_0]
    // CHECK-NEXT: // late: []
    // CHECK-NEXT: return;
    mir! {
        let borrowed: u32;
        let pointer: *const u32;
        {
            StorageLive(borrowed);
            borrowed = 2;
            pointer = &raw const borrowed;
            RET = borrowed;
            StorageDead(borrowed);
            StorageLive(borrowed);
            borrowed = 3;
            StorageDead(borrowed);
            Return()
        }
    }
}

// Partial initialization starts a lifetime, but only a bare local move ends it.
// EMIT_MIR precise_liveness.fields.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn fields() -> u32 {
    // CHECK-LABEL: fn fields(
    // CHECK: // early: []
    // CHECK-NEXT: // late: [[[AGGREGATE:_[0-9]+]]]
    // CHECK-NEXT: ([[AGGREGATE]].0: u32) = const 1_u32;
    // CHECK: = &raw const [[AGGREGATE]];
    // CHECK-NEXT: // early: [[[AGGREGATE]]]
    // CHECK-NEXT: // late: [_0, [[AGGREGATE]]]
    // CHECK-NEXT: _0 = Add(move ([[AGGREGATE]].0: u32), move ([[AGGREGATE]].1: u32));
    mir! {
        let aggregate: (u32, u32);
        let pointer: *const (u32, u32);
        {
            aggregate.0 = 1;
            aggregate.1 = 2;
            pointer = &raw const aggregate;
            RET = Move(aggregate.0) + Move(aggregate.1);
            Return()
        }
    }
}

unsafe extern "C-unwind" {
    fn callee(a: u32, b: u32) -> u32;
}

// Check the behavior around calls:
// - Destination is initialized at the late point.
// - Move arguments have their live range extended to the late point.
// - Copy arguments die at the early point when this is their last use.
// - Destination remains live on unwind edges.
// EMIT_MIR precise_liveness.call.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn call() -> u32 {
    // CHECK-LABEL: fn call(
    // CHECK: [[A:_[0-9]+]] = const 1_u32;
    // CHECK: [[B:_[0-9]+]] = const 2_u32;
    // CHECK-NEXT: // early: [[[A]], [[B]]]
    // CHECK-NEXT: // late: [[[A]], [[DESTINATION:_[0-9]+]]]
    // CHECK-NEXT: [[DESTINATION]] = callee(move [[A]], copy [[B]])
    // CHECK: = &raw const [[DESTINATION]];
    // CHECK-NEXT: // early: [[[DESTINATION]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = move [[DESTINATION]];
    // CHECK: (cleanup): {
    // CHECK-NEXT: // early: [[[DESTINATION]]]
    // CHECK-NEXT: // late: [[[DESTINATION]]]
    // CHECK-NEXT: resume;
    mir! {
        let a: u32;
        let b: u32;
        let destination: u32;
        let pointer: *const u32;
        {
            a = 1;
            b = 2;
            Call(destination = callee(Move(a), b), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            pointer = &raw const destination;
            RET = Move(destination);
            Return()
        }
        cleanup(cleanup) = {
            UnwindResume()
        }
    }
}

// A value used on only one branch is not live on the other branch.
// EMIT_MIR precise_liveness.branch.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn branch(condition: bool) -> u32 {
    // CHECK-LABEL: fn branch(
    // CHECK: [[VALUE:_[0-9]+]] = const 1_u32;
    // CHECK-NEXT: // early: [[[CONDITION:_[0-9]+]], [[VALUE]]]
    // CHECK-NEXT: // late: [[[VALUE]]]
    // CHECK-NEXT: switchInt(copy [[CONDITION]]) -> [1: [[USE:bb[0-9]+]], otherwise: [[SKIP:bb[0-9]+]]];
    // CHECK: [[USE]]: {
    // CHECK-NEXT: // early: [[[VALUE]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = move [[VALUE]];
    // CHECK: [[SKIP]]: {
    // CHECK-NEXT: // early: []
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = const 0_u32;
    mir! {
        let value: u32;
        {
            value = 1;
            match condition { true => use_value, _ => skip_value }
        }
        use_value = {
            RET = Move(value);
            Return()
        }
        skip_value = {
            RET = 0;
            Return()
        }
    }
}

// Dead destinations are live only at the late point of the assignment or call.
// EMIT_MIR precise_liveness.dead_destinations.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn dead_destinations() -> u32 {
    // CHECK-LABEL: fn dead_destinations(
    // CHECK: // early: []
    // CHECK-NEXT: // late: [[[ASSIGNMENT:_[0-9]+]]]
    // CHECK-NEXT: [[ASSIGNMENT]] = const 1_u32;
    // CHECK-NEXT: // early: []
    // CHECK-NEXT: // late: [[[DESTINATION:_[0-9]+]]]
    // CHECK-NEXT: [[DESTINATION]] = callee(const 1_u32, const 2_u32) -> [return: [[DONE:bb[0-9]+]], unwind: [[CLEANUP:bb[0-9]+]]];
    // CHECK: [[DONE]]: {
    // CHECK-NEXT: // early: []
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = const 0_u32;
    // CHECK: [[CLEANUP]] (cleanup): {
    // CHECK-NEXT: // early: []
    // CHECK-NEXT: // late: []
    // CHECK-NEXT: resume;
    mir! {
        let assignment: u32;
        let destination: u32;
        {
            assignment = 1;
            Call(destination = callee(1, 2), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = 0;
            Return()
        }
        cleanup(cleanup) = {
            UnwindResume()
        }
    }
}

// An indexed write allocates the array. The index is a use, not a destination.
// EMIT_MIR precise_liveness.indexed.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn indexed(index: usize) {
    // CHECK-LABEL: fn indexed(
    // CHECK: // early: [[[INDEX:_[0-9]+]]]
    // CHECK-NEXT: // late: [[[ARRAY:_[0-9]+]]]
    // CHECK-NEXT: [[ARRAY]][[[INDEX]]] = const 1_u32;
    mir! {
        let array: [u32; 2];
        {
            array[index] = 1;
            Return()
        }
    }
}

// A write through a pointer is a use of the pointer, not a def.
// EMIT_MIR precise_liveness.dereference.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn dereference(pointer: *mut u32) {
    // CHECK-LABEL: fn dereference(
    // CHECK: // early: [[[POINTER:_[0-9]+]]]
    // CHECK-NEXT: // late: []
    // CHECK-NEXT: (*[[POINTER]]) = const 1_u32;
    mir! {
        {
            *pointer = 1;
            Return()
        }
    }
}

// A field move keeps the base allocation live through the call's late point.
// EMIT_MIR precise_liveness.call_field.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn call_field() -> u32 {
    // CHECK-LABEL: fn call_field(
    // CHECK: ([[AGGREGATE:_[0-9]+]].0: u32) = const 1_u32;
    // CHECK-NEXT: // early: [[[AGGREGATE]]]
    // CHECK-NEXT: // late: [_0, [[AGGREGATE]]]
    // CHECK-NEXT: _0 = callee(move ([[AGGREGATE]].0: u32), const 0_u32)
    // CHECK: // early: [_0]
    // CHECK-NEXT: // late: []
    // CHECK-NEXT: return;
    mir! {
        let aggregate: (u32,);
        {
            aggregate.0 = 1;
            Call(RET = callee(Move(aggregate.0), 0), ReturnTo(done), UnwindUnreachable())
        }
        done = {
            Return()
        }
    }
}

// A borrowed field's base remains live in both successors after a field move.
// EMIT_MIR precise_liveness.call_borrowed_field.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn call_borrowed_field() -> u32 {
    // CHECK-LABEL: fn call_borrowed_field(
    // CHECK: = &raw const [[AGGREGATE:_[0-9]+]];
    // CHECK-NEXT: // early: [[[AGGREGATE]]]
    // CHECK-NEXT: // late: [[[AGGREGATE]], [[DESTINATION:_[0-9]+]]]
    // CHECK-NEXT: [[DESTINATION]] = callee(move ([[AGGREGATE]].0: u32), const 0_u32) -> [return: [[DONE:bb[0-9]+]], unwind: [[CLEANUP:bb[0-9]+]]];
    // CHECK: [[DONE]]: {
    // CHECK-NEXT: // early: [[[AGGREGATE]], [[DESTINATION]]]
    // CHECK-NEXT: // late: [_0, [[AGGREGATE]]]
    // CHECK-NEXT: _0 = move [[DESTINATION]];
    // CHECK: [[CLEANUP]] (cleanup): {
    // CHECK-NEXT: // early: [[[AGGREGATE]]]
    // CHECK-NEXT: // late: [[[AGGREGATE]]]
    // CHECK-NEXT: resume;
    mir! {
        let aggregate: (u32,);
        let destination: u32;
        let borrow: *const (u32,);
        {
            aggregate.0 = 1;
            borrow = &raw const aggregate;
            Call(destination = callee(Move(aggregate.0), 0), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = Move(destination);
            Return()
        }
        cleanup(cleanup) = {
            UnwindResume()
        }
    }
}

// An indirect move does not extend the pointer local's lifetime to the late point.
// EMIT_MIR precise_liveness.call_indirect.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn call_indirect(pointer: *const u32) -> u32 {
    // CHECK-LABEL: fn call_indirect(
    // CHECK: // early: [[[POINTER:_[0-9]+]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = callee(move (*[[POINTER]]), const 0_u32)
    mir! {
        {
            Call(RET = callee(Move(*pointer), 0), ReturnTo(done), UnwindUnreachable())
        }
        done = {
            Return()
        }
    }
}
