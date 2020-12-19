// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// #71602: check that there is no pointer comparison generated for slice equality

// CHECK-LABEL: @is_zero_slice
#[no_mangle]
pub fn is_zero_slice(data: &[u8; 4]) -> bool {
    // CHECK-NOT: %{{.+}} = icmp eq [4 x i8]* {{.+}}
    *data == [0; 4]
}
