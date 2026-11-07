//@ compile-flags: -Zmir-opt-level=0 -C opt-level=0

// EMIT_MIR index_array_and_slice.index_array.built.after.mir
fn index_array(array: &[i32; 7], index: usize) -> &i32 {
    // CHECK: bb0:
    // CHECK: _3 = copy _2;
    // CHECK: [[LT:_.+]] = Lt(copy _3, const 7_usize);
    // CHECK: assert(move [[LT]], "index out of bounds{{.+}}", const 7_usize, copy _3) -> [success: bb1, unwind

    // CHECK: bb1:
    // CHECK: _5 = &(*_1)[_3];
    // CHECK: _0 = &(*_5);
    &array[index]
}

// EMIT_MIR index_array_and_slice.index_const_generic_array.built.after.mir
fn index_const_generic_array<const N: usize>(array: &[i32; N], index: usize) -> &i32 {
    // CHECK: bb0:
    // CHECK: _3 = copy _2;
    // CHECK: [[LT:_.+]] = Lt(copy _3, const N);
    // CHECK: assert(move [[LT]], "index out of bounds{{.+}}", const N, copy _3) -> [success: bb1, unwind

    // CHECK: bb1:
    // CHECK: _5 = &(*_1)[_3];
    // CHECK: _0 = &(*_5);
    &array[index]
}

// EMIT_MIR index_array_and_slice.index_slice.built.after.mir
fn index_slice(slice: &[i32], index: usize) -> &i32 {
    // CHECK: bb0:
    // CHECK: _3 = copy _2;
    // CHECK: [[LEN:_.+]] = PtrMetadata(copy _1);
    // CHECK: [[LT:_.+]] = Lt(copy _3, copy [[LEN]]);
    // CHECK: assert(move [[LT]], "index out of bounds{{.+}}", move [[LEN]], copy _3) -> [success: bb1,

    // CHECK: bb1:
    // CHECK: _6 = &(*_1)[_3];
    // CHECK: _0 = &(*_6);
    &slice[index]
}

// EMIT_MIR index_array_and_slice.index_mut_slice.built.after.mir
fn index_mut_slice(slice: &mut [i32], index: usize) -> &i32 {
    // While the filecheck here is identical to the above test, the emitted MIR is different.
    // This cannot `copy _1` in the *built* MIR, only in the *runtime* MIR.

    // CHECK: bb0:
    // CHECK: _3 = copy _2;
    // CHECK: _4 = &raw const (fake) (*_1);
    // CHECK: [[LEN:_.+]] = PtrMetadata(move _4);
    // CHECK: [[LT:_.+]] = Lt(copy _3, copy [[LEN]]);
    // CHECK: assert(move [[LT]], "index out of bounds{{.+}}", move [[LEN]], copy _3) -> [success: bb1,

    // CHECK: bb1:
    // CHECK: _7 = &(*_1)[_3];
    // CHECK: _0 = &(*_7);
    &slice[index]
}

struct WithSliceTail(f64, [i32]);

// EMIT_MIR index_array_and_slice.index_custom.built.after.mir
fn index_custom(custom: &WithSliceTail, index: usize) -> &i32 {
    // CHECK: bb0:
    // CHECK: _3 = copy _2;
    // CHECK: [[PTR:_.+]] = &raw const (fake) ((*_1).1: [i32]);
    // CHECK: [[LEN:_.+]] = PtrMetadata(move [[PTR]]);
    // CHECK: [[LT:_.+]] = Lt(copy _3, copy [[LEN]]);
    // CHECK: assert(move [[LT]], "index out of bounds{{.+}}", move [[LEN]], copy _3) -> [success: bb1,

    // CHECK: bb1:
    // CHECK: _7 = &((*_1).1: [i32])[_3];
    // CHECK: _0 = &(*_7);
    &custom.1[index]
}

fn main() {
    index_array(&[1, 2, 3, 4, 5, 6, 7], 3);
    index_slice(&[1, 2, 3, 4, 5, 6, 7][..], 3);
    _ = index_custom;
}
