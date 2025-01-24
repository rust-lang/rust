//@ compile-flags: -C opt-level=3
//@ only-x86_64

#![crate_type = "lib"]
#![feature(iter_repeat_n)]
#![feature(array_repeat)]

#[derive(Clone)]
pub struct NotCopy(u16);

impl Drop for NotCopy {
    fn drop(&mut self) {}
}

// For a type where `Drop::drop` doesn't do anything observable and a clone is the
// same as a move, make sure that the extra case for the last item disappears.

#[no_mangle]
// CHECK-LABEL: @iter_repeat_n_next
pub fn iter_repeat_n_next(it: &mut std::iter::RepeatN<NotCopy>) -> Option<NotCopy> {
    // CHECK-NEXT: start:
    // CHECK-NOT: br
    // CHECK: %[[COUNT:.+]] = load i64
    // CHECK-NEXT: %[[COUNT_ZERO:.+]] = icmp eq i64 %[[COUNT]], 0
    // CHECK-NEXT: br i1 %[[COUNT_ZERO]], label %[[EMPTY:.+]], label %[[NOT_EMPTY:.+]]

    // CHECK: [[NOT_EMPTY]]:
    // CHECK-NEXT: %[[DEC:.+]] = add i64 %[[COUNT]], -1
    // CHECK-NEXT: store i64 %[[DEC]]
    // CHECK-NOT: br
    // CHECK: %[[VAL:.+]] = load i16
    // CHECK-NEXT: br label %[[EMPTY]]

    // CHECK: [[EMPTY]]:
    // CHECK-NOT: br
    // CHECK: phi i16
    // CHECK-SAME: [ %[[VAL]], %[[NOT_EMPTY]] ]
    // CHECK-NOT: br
    // CHECK: ret

    it.next()
}

// And as a result, using the iterator can optimize without special cases for
// the last iteration, like `memset`ing all the items in one call.

#[no_mangle]
// CHECK-LABEL: @vec_extend_via_iter_repeat_n
pub fn vec_extend_via_iter_repeat_n() -> Vec<u8> {
    // CHECK: %[[ADDR:.+]] = tail call {{(noalias )?}}noundef dereferenceable_or_null(1234) ptr @__rust_alloc(i64 noundef {{(range\(i64 1, 0\) )?}}1234, i64 noundef {{(range\(i64 1, -9223372036854775807\) )?}}1)
    // CHECK: tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(1234) %[[ADDR]], i8 42, i64 1234,

    let n = 1234_usize;
    let mut v = Vec::with_capacity(n);
    v.extend(std::iter::repeat_n(42_u8, n));
    v
}

// Array repeat uses `RepeatN::next_unchecked` internally,
// so also check that the distinction disappears there.

#[no_mangle]
// CHECK-LABEL: @array_repeat_not_copy
pub unsafe fn array_repeat_not_copy(item: NotCopy) -> [NotCopy; 8] {
    // CHECK: insertelement {{.+}} i16 %item
    // CHECK: shufflevector <8 x i16> {{.+}} zeroinitializer
    // CHECK: store <8 x i16>
    std::array::repeat(item)
}
