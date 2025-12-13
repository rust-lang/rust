//@ compile-flags: -Copt-level=3
//@ edition: 2024

#![crate_type = "lib"]

// Similar to how we `assume` that `slice::Iter::position` is within the length,
// check that `count` also does that for `TrustedLen` iterators.
// See https://rust-lang.zulipchat.com/#narrow/channel/122651-general/topic/Overflow-chk.20removed.20for.20array.20of.2059.2C.20but.20not.2060.2C.20elems/with/561070780

// CHECK-LABEL: @filter_count_untrusted
#[unsafe(no_mangle)]
pub fn filter_count_untrusted(bar: &[u8; 1234]) -> u16 {
    // CHECK-NOT: llvm.assume
    // CHECK: call void @{{.+}}unwrap_failed
    // CHECK-NOT: llvm.assume
    let mut iter = bar.iter();
    let iter = std::iter::from_fn(|| iter.next()); // Make it not TrustedLen
    u16::try_from(iter.filter(|v| **v == 0).count()).unwrap()
}

// CHECK-LABEL: @filter_count_trusted
#[unsafe(no_mangle)]
pub fn filter_count_trusted(bar: &[u8; 1234]) -> u16 {
    // CHECK-NOT: unwrap_failed
    // CHECK: %[[ASSUME:.+]] = icmp ult {{i64|i32|i16}} %{{.+}}, 1235
    // CHECK-NEXT: tail call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NOT: unwrap_failed
    let iter = bar.iter();
    u16::try_from(iter.filter(|v| **v == 0).count()).unwrap()
}

// CHECK: ; core::result::unwrap_failed
// CHECK-NEXT: Function Attrs
// CHECK-NEXT: declare{{.+}}void @{{.+}}unwrap_failed
