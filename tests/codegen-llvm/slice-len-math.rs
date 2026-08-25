//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: @len_plus_ten_a
pub fn len_plus_ten_a(s: &[u8]) -> usize {
    // CHECK: start:
    // CHECK-NOT: add
    // CHECK: %[[R:.+]] = add nuw i{{.+}} %s.1, 10
    // CHECK-NEXT: ret {{.+}} %[[R]]
    s.len().wrapping_add(10)
}

#[no_mangle]
// CHECK-LABEL: @len_plus_ten_b
pub fn len_plus_ten_b(s: &[u32]) -> usize {
    // CHECK: start:
    // CHECK-NOT: add
    // CHECK: %[[R:.+]] = add nuw nsw i{{.+}} %s.1, 10
    // CHECK-NEXT: ret {{.+}} %[[R]]
    s.len().wrapping_add(10)
}

#[no_mangle]
// CHECK-LABEL: @len_plus_len
pub fn len_plus_len(x: &[u8], y: &[u8]) -> usize {
    // CHECK: start:
    // CHECK-NOT: add
    // CHECK: %[[R:.+]] = add nuw i{{.+}} {{%x.1, %y.1|%y.1, %x.1}}
    // CHECK-NEXT: ret {{.+}} %[[R]]
    usize::wrapping_add(x.len(), y.len())
}
