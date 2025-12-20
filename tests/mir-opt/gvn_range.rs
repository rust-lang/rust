//@ test-mir-pass: GVN

#![crate_type = "lib"]

#[repr(i8)]
pub enum SignedWrapping {
    Start = i8::MAX,
    End = i8::MIN,
}

#[repr(u8)]
pub enum UnsignedWrapping {
    Start = u8::MAX,
    End = u8::MIN,
}

#[repr(u16)]
pub enum UnsignedU16 {
    Start = 3,
    End = 4,
}

#[repr(i8)]
#[derive(Copy, Clone)]
pub enum SignedA {
    A = -2,
    B = -1,
    C = 0,
    D = 1,
}

// EMIT_MIR gvn_range.cast_to_unsigned_wrapping.GVN.diff
pub fn cast_to_unsigned_wrapping(s: SignedA) -> bool {
    // The source range of i8 is [-2, 1], and the destination range of u8 is [254, 1].
    // This is a wrapping range that contain 255.
    // CHECK-LABEL: fn cast_to_unsigned_wrapping(
    // CHECK: _0 = Gt(
    s as u8 > 254
}

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum UnsignedA {
    A = 127,
    B = 128, // -128, 0x80
    C = 129, // -127, 0x81
    D = 130, // -126, 0x82
}

// EMIT_MIR gvn_range.cast_to_signed_wrapping.GVN.diff
pub fn cast_to_signed_wrapping(s: UnsignedA) -> bool {
    // The source range of u8 is [127, 130], and the destination range of i8 is [127, -126].
    // This is a wrapping range that contain -127.
    // CHECK-LABEL: fn cast_to_signed_wrapping(
    // CHECK: _0 = Gt(
    s as i8 > -126
}

// The size of range [0xff, 1] is smaller than [0, 0xff].
#[repr(u8)]
#[derive(Copy, Clone)]
pub enum UnsignedWrappingA {
    A = u8::MAX,     // -1, 0xff
    B = u8::MIN,     // 0
    C = u8::MIN + 1, // 1
}

// EMIT_MIR gvn_range.cast_from_unsigned_wrapping.GVN.diff
pub fn cast_from_unsigned_wrapping(s: UnsignedWrappingA) -> (bool, bool) {
    // CHECK-LABEL: fn cast_from_unsigned_wrapping(
    // CHECK: _0 = const (false, false);
    let s = s as i8;
    (s < -1, s > 1)
}

// The size of range [0, -128] is smaller than [-128, 127].
#[repr(i8)]
#[derive(Copy, Clone)]
pub enum SignedWrappingA {
    A = i8::MAX, // 127, 0x7f
    B = i8::MIN, // -128, 0x80
    C = 0,
}

// EMIT_MIR gvn_range.cast_from_signed_wrapping.GVN.diff
pub fn cast_from_signed_wrapping(s: SignedWrappingA) -> (bool, bool) {
    // CHECK-LABEL: fn cast_from_signed_wrapping(
    // CHECK: _0 = const (true, false);
    let s = s as u8;
    (s <= 128, s > 128)
}

// EMIT_MIR gvn_range.unsigned_extension.GVN.diff
pub fn unsigned_extension(u: UnsignedA) -> (bool, bool, bool, bool) {
    // CHECK-LABEL: fn unsigned_extension(
    // CHECK: _0 = (const false, const false, const false, const false);
    let u2u = u as u16;
    let u2s = u as i16;
    (u2u < 127, u2u > 130, u2s < 127, u2u > 130)
}

// EMIT_MIR gvn_range.unsigned_wrapping_extension.GVN.diff
pub fn unsigned_wrapping_extension(u: UnsignedWrappingA) -> (bool, bool) {
    // CHECK-LABEL: fn unsigned_wrapping_extension(
    // CHECK: _0 = const (true, true);
    let u2u = u as u16;
    let u2s = u as i16;
    (u2u <= 256, u2s <= 256)
}

// EMIT_MIR gvn_range.signed_extension.GVN.diff
pub fn signed_extension(s: SignedA) -> (bool, bool, bool, bool) {
    // CHECK-LABEL: fn signed_extension(
    // CHECK: _0 = (move _{{.*}}, move _{{.*}}, const false, const false);
    let s2u = s as u16;
    let s2s = s as i16;
    (s2u > 254, s2u > 65534, s2s < -2, s2s > 1)
}

// EMIT_MIR gvn_range.signed_wrapping_extension.GVN.diff
pub fn signed_wrapping_extension(s: SignedWrappingA) -> (bool, bool, bool, bool) {
    // CHECK-LABEL: fn signed_wrapping_extension(
    // CHECK: _0 = (move _{{.*}}, move _{{.*}}, const false, const false);
    let s2u = s as u16;
    let s2s = s as i16;
    (s2u < 128, s2u > 65408, s2s < -128, s2s > 128)
}

#[repr(i16)]
pub enum SignedWrappingB {
    A = 32512,  // 0x7f00u16
    B = 32767,  // 0x7fffu16
    C = -32768, // 0x8000u16
}

#[repr(i16)]
pub enum SignedWrappingC {
    A = 256, // 0x100
    B = 511, // 0x1ff
    C = i16::MAX,
    D = i16::MIN,
    X = 0,
}

#[repr(i16)]
pub enum SignedB {
    A = 0,
    B = 255, // 0xff
    C = 256, // 0x100
}

// When cast to i8 or u8,
// the ranges [0x7f00, 0x8000], [0x100, 0] and [0, 0x100] contain the value -1.
// EMIT_MIR gvn_range.truncate.GVN.diff
pub fn truncate(s1: SignedWrappingB, s2: SignedWrappingC, s3: SignedB) -> (bool, bool, bool) {
    // CHECK-LABEL: fn truncate(
    // CHECK: _0 = (move _{{.*}}, move _{{.*}}, move _{{.*}});
    (s1 as i8 > -1, s2 as i8 > -1, s3 as i8 > -1)
}
