//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

const JSVAL_TAG_CLEAR: u32 = 0xFFFFFF80;
const JSVAL_TYPE_INT32: u8 = 0x01;
const JSVAL_TYPE_UNDEFINED: u8 = 0x02;
#[repr(u32)]
enum ValueTag {
    JSVAL_TAG_INT32 = JSVAL_TAG_CLEAR | (JSVAL_TYPE_INT32 as u32),
    JSVAL_TAG_UNDEFINED = JSVAL_TAG_CLEAR | (JSVAL_TYPE_UNDEFINED as u32),
}

fn main() {
    let _ = ValueTag::JSVAL_TAG_INT32;
}
