//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// CHECK-LABEL: @write_u8_variant_a
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT: icmp ugt
// CHECK-NEXT: getelementptr
// CHECK-NEXT: select i1 {{.+}} null
// CHECK-NEXT: insertvalue
// CHECK-NEXT: insertvalue
// CHECK-NEXT: ret
#[no_mangle]
pub fn write_u8_variant_a(bytes: &mut [u8], buf: u8, offset: usize) -> Option<&mut [u8]> {
    let buf = buf.to_le_bytes();
    bytes.get_mut(offset..).and_then(|bytes| bytes.get_mut(..buf.len()))
}
