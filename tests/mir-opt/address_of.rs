//@ test-mir-pass: SimplifyCfg-pre-optimizations
// EMIT_MIR address_of.address_of_reborrow.SimplifyCfg-initial.after.mir

// CHECK: fn address_of_reborrow()
// CHECK-NOT: }
fn address_of_reborrow() {
    // CHECK-DAG: {{^ *}} debug y => [[y:_[0-9]+]]
    let y = &[0; 10];
    // CHECK-DAG: {{^ *}} debug z => [[z:_[0-9]+]]
    let mut z = &mut [0; 10];

    y as *const _;
    y as *const [i32; 10];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp1:_[0-9]+]] as *const dyn std::marker::Send (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}} [[tmp1]] = &raw const (*[[y]]);
    y as *const dyn Send;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp2:_[0-9]+]] as *const [i32] (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}}[[tmp2]] = &raw const (*[[y]]);
    y as *const [i32];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp3:_[0-9]+]] as *const i32 (PtrToPtr);
    // CHECK-DAG: {{^ *}}[[tmp3]] = &raw const (*[[y]]);
    y as *const i32; // This is a cast, not a coercion

    let p: *const _ = y;
    let p: *const [i32; 10] = y;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp4:_[0-9]+]] as *const dyn std::marker::Send (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp4]] = &raw const (*[[y]]);
    let p: *const dyn Send = y;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp5:_[0-9]+]] as *const [i32] (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp5]] = &raw const (*[[y]]);
    let p: *const [i32] = y;

    z as *const _;
    z as *const [i32; 10];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp6:_[0-9]+]] as *const dyn std::marker::Send (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}}[[tmp6]] = &raw const (*[[z]]);
    z as *const dyn Send;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp7:_[0-9]+]] as *const [i32] (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}}[[tmp7]] = &raw const (*[[z]]);
    z as *const [i32];

    let p: *const _ = z;
    let p: *const [i32; 10] = z;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp8:_[0-9]+]] as *const dyn std::marker::Send (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp8]] = &raw const (*[[z]]);
    let p: *const dyn Send = z;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp9:_[0-9]+]] as *const [i32] (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp9]] = &raw const (*[[z]]);
    let p: *const [i32] = z;

    z as *mut _;
    z as *mut [i32; 10];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp10:_[0-9]+]] as *mut dyn std::marker::Send (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}}[[tmp10]] = &raw mut (*[[z]]);
    z as *mut dyn Send;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp11:_[0-9]+]] as *mut [i32] (PointerCoercion(Unsize, AsCast));
    // CHECK-DAG: {{^ *}}[[tmp11]] = &raw mut (*[[z]]);
    z as *mut [i32];

    let p: *mut _ = z;
    let p: *mut [i32; 10] = z;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp12:_[0-9]+]] as *mut dyn std::marker::Send (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp12]] = &raw mut (*[[z]]);
    let p: *mut dyn Send = z;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = move [[tmp13:_[0-9]+]] as *mut [i32] (PointerCoercion(Unsize, Implicit));
    // CHECK-DAG: {{^ *}}[[tmp13]] = &raw mut (*[[z]]);
    let p: *mut [i32] = z;
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[y]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[y]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[y]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[y]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw mut (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw mut (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw mut (*[[z]]);
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw mut (*[[z]]);
}

// The normal borrows here should be preserved
// EMIT_MIR address_of.borrow_and_cast.SimplifyCfg-initial.after.mir
//
// CHECK: fn borrow_and_cast([[x:_[0-9]+]]: i32)
// CHECK-NOT: }
fn borrow_and_cast(mut x: i32) {
    // CHECK-DAG: {{^ *}}let [[p:_[0-9]+]]: &i32;
    let p = &x as *const i32;
    // CHECK-DAG: {{^ *}}let [[q:_[0-9]+]]: &mut i32;
    let q = &mut x as *const i32;
    // CHECK-DAG: {{^ *}}let mut [[r:_[0-9]+]]: &mut i32;
    let r = &mut x as *mut i32;
    // CHECK-DAG: {{^ *}}[[p]] = &[[x]];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[p]]);
    // CHECK-DAG: {{^ *}}[[q]] = &mut [[x]];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw const (*[[q]]);
    // CHECK-DAG: {{^ *}}[[r]] = &mut [[x]];
    // CHECK-DAG: {{^ *}} _{{[0-9]+}} = &raw mut (*[[r]]);
}

fn main() {}
