//@ compile-flags: -O
#![crate_type = "lib"]

// CHECK-LABEL: @write_u8_variant_a
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT: icmp ugt
// CHECK-NEXT: getelementptr
// CHECK-NEXT: select i1 {{.+}}
// CHECK-NEXT: select i1 {{.+}}
// CHECK-NEXT: insertvalue
// CHECK-NEXT: insertvalue
// CHECK-NEXT: ret
#[no_mangle]
pub fn write_u8_variant_a(bytes: &mut [u8], buf: u8, offset: usize) -> Option<&mut [u8]> {
    let buf = buf.to_le_bytes();
    bytes.get_mut(offset..).and_then(|bytes| bytes.get_mut(..buf.len()))
}
