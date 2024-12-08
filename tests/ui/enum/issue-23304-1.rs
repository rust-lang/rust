//@ run-pass
#![allow(dead_code)]

#[repr(u8)]
#[allow(dead_code)]
enum ValueType {
    DOUBLE              = 0x00,
    INT32               = 0x01,
}

#[repr(u32)]
enum ValueTag {
    INT32                = 0x1FFF0u32 | (ValueType::INT32 as u32),
    X,
}

#[repr(u64)]
enum ValueShiftedTag {
    INT32        = ValueTag::INT32 as u64,
    X,
}

fn main() {
    println!("{}", ValueTag::INT32 as u32);
}
