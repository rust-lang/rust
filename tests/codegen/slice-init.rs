// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK-LABEL: @zero_sized_elem
#[no_mangle]
pub fn zero_sized_elem() {
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0
    let x = [(); 4];
    opaque(&x);
}

// CHECK-LABEL: @zero_len_array
#[no_mangle]
pub fn zero_len_array() {
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0
    let x = [4; 0];
    opaque(&x);
}

// CHECK-LABEL: @byte_array
#[no_mangle]
pub fn byte_array() {
    // CHECK: call void @llvm.memset.{{.+}}({{i8\*|ptr}} {{.*}}, i8 7, i{{[0-9]+}} 4
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [7u8; 4];
    opaque(&x);
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
enum Init {
    Loop,
    Memset,
}

// CHECK-LABEL: @byte_enum_array
#[no_mangle]
pub fn byte_enum_array() {
    // CHECK: call void @llvm.memset.{{.+}}({{i8\*|ptr}} {{.*}}, i8 {{.*}}, i{{[0-9]+}} 4
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [Init::Memset; 4];
    opaque(&x);
}

// CHECK-LABEL: @zeroed_integer_array
#[no_mangle]
pub fn zeroed_integer_array() {
    // CHECK: call void @llvm.memset.{{.+}}({{i8\*|ptr}} {{.*}}, i8 0, i{{[0-9]+}} 16
    // CHECK-NOT: br label %repeat_loop_header{{.*}}
    let x = [0u32; 4];
    opaque(&x);
}

// CHECK-LABEL: @nonzero_integer_array
#[no_mangle]
pub fn nonzero_integer_array() {
    // CHECK: br label %repeat_loop_header{{.*}}
    // CHECK-NOT: call void @llvm.memset.p0
    let x = [0x1a_2b_3c_4d_u32; 4];
    opaque(&x);
}

// Use an opaque function to prevent rustc from removing useless drops.
#[inline(never)]
pub fn opaque(_: impl Sized) {}
