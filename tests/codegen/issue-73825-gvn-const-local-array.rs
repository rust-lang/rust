// issue: <https://github.com/rust-lang/rust/issues/73825>
//@ compile-flags: -C opt-level=1
#![crate_type = "lib"]

// CHECK-LABEL: @foo
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT: and
// CHECK-NEXT: getelementptr inbounds
// CHECK-NEXT: load i32
// CHECK-NEXT: ret i32
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
