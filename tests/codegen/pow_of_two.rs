// compile-flags: --crate-type=lib -Zmerge-functions=disabled -O -C overflow-checks=false

// CHECK-LABEL: @a(
#[no_mangle]
pub fn a(exp: u32) -> u64 {
    // CHECK: %{{[^ ]+}} = icmp ugt i32 %exp, 64
    // CHECK: %{{[^ ]+}} = zext i32 %exp to i64
    // CHECK: %{{[^ ]+}} = shl nuw i64 {{[^ ]+}}, %{{[^ ]+}}
    // CHECK: ret i64 %{{[^ ]+}}
    2u64.pow(exp)
}

// CHECK-LABEL: @b(
#[no_mangle]
pub fn b(exp: u32) -> i64 {
    // CHECK: %{{[^ ]+}} = icmp ugt i32 %exp, 64
    // CHECK: %{{[^ ]+}} = zext i32 %exp to i64
    // CHECK: %{{[^ ]+}} = shl nuw i64 {{[^ ]+}}, %{{[^ ]+}}
    // CHECK: ret i64 %{{[^ ]+}}
    2i64.pow(exp)
}

// CHECK-LABEL: @c(
#[no_mangle]
pub fn c(exp: u32) -> u32 {
    // CHECK: %{{[^ ]+}} = icmp ugt i32 %exp, 16
    // CHECK: %{{[^ ]+}} = shl nuw nsw i32 %exp, 1
    // CHECK: %{{[^ ]+}} = shl nuw i32 1, %{{[^ ]+}}
    // CHECK: %{{[^ ]+}} = select i1 %{{[^ ]+}}, i32 0, i32 %{{[^ ]+}}
    // CHECK: ret i32 %{{[^ ]+}}
    4u32.pow(exp)
}

// CHECK-LABEL: @d(
#[no_mangle]
pub fn d(exp: u32) -> u32 {
    // CHECK: %{{[^ ]+}} = icmp ugt i32 %exp, 6
    // CHECK: %{{[^ ]+}} = mul nuw nsw i32 %exp, 5
    // CHECK: %{{[^ ]+}} = shl nuw nsw i32 1, %{{[^ ]+}}
    // CHECK: %{{[^ ]+}} = select i1 {{[^ ]+}}, i32 0, i32 %{{[^ ]+}}
    // CHECK: ret i32 %{{[^ ]+}}
    32u32.pow(exp)
}

// CHECK-LABEL: @e(
#[no_mangle]
pub fn e(exp: u32) -> i32 {
    // CHECK: %{{[^ ]+}} = icmp ugt i32 %exp, 6
    // CHECK: %{{[^ ]+}} = mul nuw {{(nsw )?}}i32 %exp, 5
    // CHECK: %{{[^ ]+}} = shl nuw {{(nsw )?}}i32 1, %{{[^ ]+}}
    // CHECK: %{{[^ ]+}} = select i1 {{[^ ]+}}, i32 0, i32 %{{[^ ]+}}
    // CHECK: ret i32 %{{[^ ]+}}
    32i32.pow(exp)
}
// note: d and e are expected to yield the same IR
