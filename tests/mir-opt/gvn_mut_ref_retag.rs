//@ test-mir-pass: GVN
//@ compile-flags: -Zmir-opt-level=1 -Cpanic=abort

// EMIT_MIR gvn_mut_ref_retag.main.GVN.diff

fn use_mut_ref(value: &mut String) {
    value.push_str(" world");
}

fn borrow_mut_ref(value: &&mut String) {
    let _value = &**value;
}

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: [[SOURCE:_.*]] = &mut (*_2);
    // CHECK: [[REFERENCE:_.*]] = copy [[SOURCE]];
    // CHECK: _6 = use_mut_ref(copy [[REFERENCE]])
    let mut value = String::from("hello");
    let pointer = &mut value as *mut String;

    let reference = unsafe { &mut *pointer };
    use_mut_ref(reference);
    borrow_mut_ref(&reference);
}
