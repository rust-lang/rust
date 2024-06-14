// issue: <https://github.com/rust-lang/rust/issues/73825>
//@ compile-flags: -C opt-level=1
#![crate_type = "lib"]

// CHECK-LABEL: @foo
// CHECK-NEXT: start:
// CHECK-NEXT: %_3 = and i64 %x, 63
// CHECK-NEXT: %0 = getelementptr inbounds [64 x i32], ptr @0, i64 0, i64 %_3
// CHECK-NEXT: %_0 = load i32, ptr %0, align 4
// CHECK-NEXT: ret i32 %_0
#[no_mangle]
#[rustfmt::skip]
pub fn foo(x: usize) -> i32 {
    let base: [i32; 64] = [
        67, 754, 860, 559, 368, 870, 548, 972,
        141, 731, 351, 664, 32, 4, 996, 741,
        203, 292, 237, 480, 151, 940, 777, 540,
        143, 587, 747, 65, 152, 517, 882, 880,
        712, 595, 370, 901, 237, 53, 789, 785,
        912, 650, 896, 367, 316, 392, 62, 473,
        675, 691, 281, 192, 445, 970, 225, 425,
        628, 324, 322, 206, 912, 867, 462, 92
    ];
    base[x % 64]
}

// This checks whether LLVM de-duplicates `promoted` array and `base` array.
// Because in MIR, `&[..]` is already promoted by promote pass. GVN keeps promoting
// `*&[..]` to `const [..]` again.
//
// CHECK-LABEL: @deduplicability
// CHECK-NEXT: start:
// CHECK-NEXT: %_3 = and i64 %x, 63
// CHECK-NEXT: %0 = getelementptr inbounds [64 x i32], ptr @0, i64 0, i64 %_3
// CHECK-NEXT: %_0 = load i32, ptr %0, align 4
// CHECK-NEXT: ret i32 %_0
#[no_mangle]
#[rustfmt::skip]
pub fn deduplicability(x: usize) -> i32 {
    let promoted = *&[
        67i32, 754, 860, 559, 368, 870, 548, 972,
        141, 731, 351, 664, 32, 4, 996, 741,
        203, 292, 237, 480, 151, 940, 777, 540,
        143, 587, 747, 65, 152, 517, 882, 880,
        712, 595, 370, 901, 237, 53, 789, 785,
        912, 650, 896, 367, 316, 392, 62, 473,
        675, 691, 281, 192, 445, 970, 225, 425,
        628, 324, 322, 206, 912, 867, 462, 92
    ];
    promoted[x % 64]
}
