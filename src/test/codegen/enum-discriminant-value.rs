// Verify that DIEnumerator uses isUnsigned flag when appropriate.
//
// compile-flags: -g -C no-prepopulate-passes

#[repr(i64)]
pub enum I64 {
    I64Min = std::i64::MIN,
    I64Max = std::i64::MAX,
}

#[repr(u64)]
pub enum U64 {
    U64Min = std::u64::MIN,
    U64Max = std::u64::MAX,
}

fn main() {
    let _a = I64::I64Min;
    let _b = I64::I64Max;
    let _c = U64::U64Min;
    let _d = U64::U64Max;
}

// CHECK: !DIEnumerator(name: "I64Min", value: -9223372036854775808)
// CHECK: !DIEnumerator(name: "I64Max", value: 9223372036854775807)
// CHECK: !DIEnumerator(name: "U64Min", value: 0, isUnsigned: true)
// CHECK: !DIEnumerator(name: "U64Max", value: 18446744073709551615, isUnsigned: true)
