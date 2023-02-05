// unit-test: EnumSizeOpt
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// compile-flags: -Zunsound-mir-opts

#![feature(arbitrary_enum_discriminant, repr128)]

// Tests that an enum with a variant with no data gets correctly transformed.
pub enum NoData {
  Large([u8; 8196]),
  None,
}

// Tests that an enum with a variant with data that is a valid candidate gets transformed.
pub enum Candidate {
  Small(u8),
  Large([u8; 8196]),
}

// Tests that an enum which has a discriminant much higher than the variant does not get
// tformed.
#[repr(u32)]
pub enum InvalidIdxs {
  A = 302,
  Large([u64; 1024]),
}

// Tests that an enum with too high of a discriminant index (not in bounds of usize) does not
// get tformed.
#[repr(u128)]
pub enum NotTrunctable {
    A = 0,
    B([u8; 1024]) = 1,
    C([u8; 4096]) = 0x10000000000000001,
}

// Tests that an enum with discriminants in random order still gets tformed correctly.
#[repr(u32)]
pub enum RandOrderDiscr {
  A = 13,
  B([u8; 1024]) = 5,
  C = 7,
}

// EMIT_MIR enum_opt.unin.EnumSizeOpt.diff
pub fn unin() -> NoData {
  let mut a = NoData::None;
  a = NoData::Large([1; 8196]);
  a
}

// EMIT_MIR enum_opt.cand.EnumSizeOpt.diff
pub fn cand() -> Candidate {
  let mut a = Candidate::Small(1);
  a = Candidate::Large([1; 8196]);
  a
}

// EMIT_MIR enum_opt.invalid.EnumSizeOpt.diff
pub fn invalid() -> InvalidIdxs {
  let mut a = InvalidIdxs::A;
  a = InvalidIdxs::Large([0; 1024]);
  a
}

// EMIT_MIR enum_opt.trunc.EnumSizeOpt.diff
pub fn trunc() -> NotTrunctable {
  let mut a = NotTrunctable::A;
  a = NotTrunctable::B([0; 1024]);
  a = NotTrunctable::C([0; 4096]);
  a
}

pub fn rand_order() -> RandOrderDiscr {
  let mut a = RandOrderDiscr::A;
  a = RandOrderDiscr::B([0; 1024]);
  a = RandOrderDiscr::C;
  a
}

pub fn main() {
  unin();
  cand();
  invalid();
  trunc();
  rand_order();
}
