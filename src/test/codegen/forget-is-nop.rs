// compile-flags: -C opt-level=0

#![crate_type = "lib"]

// CHECK-LABEL: mem6forget{{.+}}[100 x %"std::string::String"]*
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK: ret

// CHECK-LABEL: mem6forget{{.+}}[100 x i64]*
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK: ret

pub fn forget_large_copy_type(whatever: [i64; 100]) {
    std::mem::forget(whatever)
}

pub fn forget_large_drop_type(whatever: [String; 100]) {
    std::mem::forget(whatever)
}
