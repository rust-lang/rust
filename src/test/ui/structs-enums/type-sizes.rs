// run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]
#![feature(never_type)]

use std::mem::size_of;
use std::num::NonZeroU8;

struct t {a: u8, b: i8}
struct u {a: u8, b: i8, c: u8}
struct v {a: u8, b: i8, c: v2, d: u32}
struct v2 {u: char, v: u8}
struct w {a: isize, b: ()}
struct x {a: isize, b: (), c: ()}
struct y {x: isize}

enum e1 {
    a(u8, u32), b(u32), c
}
enum e2 {
    a(u32), b
}

#[repr(C, u8)]
enum e3 {
    a([u16; 0], u8), b
}

struct ReorderedStruct {
    a: u8,
    b: u16,
    c: u8
}

enum ReorderedEnum {
    A(u8, u16, u8),
    B(u8, u16, u8),
}

enum ReorderedEnum2 {
    A(u8, u32, u8),
    B(u16, u8, u16, u8),

    // 0x100 niche variants.
    _00, _01, _02, _03, _04, _05, _06, _07, _08, _09, _0A, _0B, _0C, _0D, _0E, _0F,
    _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _1A, _1B, _1C, _1D, _1E, _1F,
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _2A, _2B, _2C, _2D, _2E, _2F,
    _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _3A, _3B, _3C, _3D, _3E, _3F,
    _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _4A, _4B, _4C, _4D, _4E, _4F,
    _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _5A, _5B, _5C, _5D, _5E, _5F,
    _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _6A, _6B, _6C, _6D, _6E, _6F,
    _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _7A, _7B, _7C, _7D, _7E, _7F,
    _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _8A, _8B, _8C, _8D, _8E, _8F,
    _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, _9A, _9B, _9C, _9D, _9E, _9F,
    _A0, _A1, _A2, _A3, _A4, _A5, _A6, _A7, _A8, _A9, _AA, _AB, _AC, _AD, _AE, _AF,
    _B0, _B1, _B2, _B3, _B4, _B5, _B6, _B7, _B8, _B9, _BA, _BB, _BC, _BD, _BE, _BF,
    _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _CA, _CB, _CC, _CD, _CE, _CF,
    _D0, _D1, _D2, _D3, _D4, _D5, _D6, _D7, _D8, _D9, _DA, _DB, _DC, _DD, _DE, _DF,
    _E0, _E1, _E2, _E3, _E4, _E5, _E6, _E7, _E8, _E9, _EA, _EB, _EC, _ED, _EE, _EF,
    _F0, _F1, _F2, _F3, _F4, _F5, _F6, _F7, _F8, _F9, _FA, _FB, _FC, _FD, _FE, _FF,
}

enum EnumEmpty {}

enum EnumSingle1 {
    A,
}

enum EnumSingle2 {
    A = 42 as isize,
}

enum EnumSingle3 {
    A,
    B(!),
}

#[repr(u8)]
enum EnumSingle4 {
    A,
}

#[repr(u8)]
enum EnumSingle5 {
    A = 42 as u8,
}

enum EnumWithMaybeUninhabitedVariant<T> {
    A(&'static ()),
    B(&'static (), T),
    C,
}

enum NicheFilledEnumWithAbsentVariant {
    A(&'static ()),
    B((), !),
    C,
}

enum Option2<A, B> {
    Some(A, B),
    None
}

// Two layouts are considered for `CanBeNicheFilledButShouldnt`:
//   Niche-filling:
//     { u32 (4 bytes), NonZeroU8 + tag in niche (1 byte), padding (3 bytes) }
//   Tagged:
//     { tag (1 byte), NonZeroU8 (1 byte), padding (2 bytes), u32 (4 bytes) }
// Both are the same size (due to padding),
// but the tagged layout is better as the tag creates a niche with 254 invalid values,
// allowing types like `Option<Option<CanBeNicheFilledButShouldnt>>` to fit into 8 bytes.
pub enum CanBeNicheFilledButShouldnt {
    A(NonZeroU8, u32),
    B
}
pub enum AlwaysTaggedBecauseItHasNoNiche {
    A(u8, u32),
    B
}

pub enum NicheFilledMultipleFields {
    A(bool, u8),
    B(u8),
    C(u8),
    D(bool),
    E,
    F,
    G,
}

struct BoolInTheMiddle(std::num::NonZeroU16, bool, u8);

enum NicheWithData {
    A,
    B([u16; 5]),
    Largest { a1: u32, a2: BoolInTheMiddle, a3: u32 },
    C,
    D(u32, u32),
}

// A type with almost 2^16 invalid values.
#[repr(u16)]
pub enum NicheU16 {
    _0,
}

pub enum EnumManyVariant<X> {
    Dataful(u8, X),

    // 0x100 niche variants.
    _00, _01, _02, _03, _04, _05, _06, _07, _08, _09, _0A, _0B, _0C, _0D, _0E, _0F,
    _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _1A, _1B, _1C, _1D, _1E, _1F,
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _2A, _2B, _2C, _2D, _2E, _2F,
    _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _3A, _3B, _3C, _3D, _3E, _3F,
    _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _4A, _4B, _4C, _4D, _4E, _4F,
    _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _5A, _5B, _5C, _5D, _5E, _5F,
    _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _6A, _6B, _6C, _6D, _6E, _6F,
    _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _7A, _7B, _7C, _7D, _7E, _7F,
    _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _8A, _8B, _8C, _8D, _8E, _8F,
    _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, _9A, _9B, _9C, _9D, _9E, _9F,
    _A0, _A1, _A2, _A3, _A4, _A5, _A6, _A7, _A8, _A9, _AA, _AB, _AC, _AD, _AE, _AF,
    _B0, _B1, _B2, _B3, _B4, _B5, _B6, _B7, _B8, _B9, _BA, _BB, _BC, _BD, _BE, _BF,
    _C0, _C1, _C2, _C3, _C4, _C5, _C6, _C7, _C8, _C9, _CA, _CB, _CC, _CD, _CE, _CF,
    _D0, _D1, _D2, _D3, _D4, _D5, _D6, _D7, _D8, _D9, _DA, _DB, _DC, _DD, _DE, _DF,
    _E0, _E1, _E2, _E3, _E4, _E5, _E6, _E7, _E8, _E9, _EA, _EB, _EC, _ED, _EE, _EF,
    _F0, _F1, _F2, _F3, _F4, _F5, _F6, _F7, _F8, _F9, _FA, _FB, _FC, _FD, _FE, _FF,
}

pub fn main() {
    assert_eq!(size_of::<u8>(), 1 as usize);
    assert_eq!(size_of::<u32>(), 4 as usize);
    assert_eq!(size_of::<char>(), 4 as usize);
    assert_eq!(size_of::<i8>(), 1 as usize);
    assert_eq!(size_of::<i32>(), 4 as usize);
    assert_eq!(size_of::<t>(), 2 as usize);
    assert_eq!(size_of::<u>(), 3 as usize);
    // Alignment causes padding before the char and the u32.

    assert_eq!(size_of::<v>(),
                16 as usize);
    assert_eq!(size_of::<isize>(), size_of::<usize>());
    assert_eq!(size_of::<w>(), size_of::<isize>());
    assert_eq!(size_of::<x>(), size_of::<isize>());
    assert_eq!(size_of::<isize>(), size_of::<y>());

    // Make sure enum types are the appropriate size, mostly
    // around ensuring alignment is handled properly

    assert_eq!(size_of::<e1>(), 8 as usize);
    assert_eq!(size_of::<e2>(), 8 as usize);
    assert_eq!(size_of::<e3>(), 4 as usize);
    assert_eq!(size_of::<ReorderedStruct>(), 4);
    assert_eq!(size_of::<ReorderedEnum>(), 6);
    assert_eq!(size_of::<ReorderedEnum2>(), 8);


    assert_eq!(size_of::<EnumEmpty>(), 0);
    assert_eq!(size_of::<EnumSingle1>(), 0);
    assert_eq!(size_of::<EnumSingle2>(), 0);
    assert_eq!(size_of::<EnumSingle3>(), 0);
    assert_eq!(size_of::<EnumSingle4>(), 1);
    assert_eq!(size_of::<EnumSingle5>(), 1);

    assert_eq!(size_of::<EnumWithMaybeUninhabitedVariant<!>>(),
               size_of::<EnumWithMaybeUninhabitedVariant<()>>());
    assert_eq!(size_of::<NicheFilledEnumWithAbsentVariant>(), size_of::<&'static ()>());

    assert_eq!(size_of::<Option<Option<(bool, &())>>>(), size_of::<(bool, &())>());
    assert_eq!(size_of::<Option<Option<(&(), bool)>>>(), size_of::<(bool, &())>());
    assert_eq!(size_of::<Option<Option2<bool, &()>>>(), size_of::<(bool, &())>());
    assert_eq!(size_of::<Option<Option2<&(), bool>>>(), size_of::<(bool, &())>());

    assert_eq!(size_of::<CanBeNicheFilledButShouldnt>(), 8);
    assert_eq!(size_of::<Option<CanBeNicheFilledButShouldnt>>(), 8);
    assert_eq!(size_of::<Option<Option<CanBeNicheFilledButShouldnt>>>(), 8);
    assert_eq!(size_of::<AlwaysTaggedBecauseItHasNoNiche>(), 8);
    assert_eq!(size_of::<Option<AlwaysTaggedBecauseItHasNoNiche>>(), 8);
    assert_eq!(size_of::<Option<Option<AlwaysTaggedBecauseItHasNoNiche>>>(), 8);

    assert_eq!(size_of::<NicheFilledMultipleFields>(), 2);
    assert_eq!(size_of::<Option<NicheFilledMultipleFields>>(), 2);
    assert_eq!(size_of::<Option<Option<NicheFilledMultipleFields>>>(), 2);

    struct S1{ a: u16, b: std::num::NonZeroU16, c: u16, d: u8, e: u32, f: u64, g:[u8;2] }
    assert_eq!(size_of::<S1>(), 24);
    assert_eq!(size_of::<Option<S1>>(), 24);

    assert_eq!(size_of::<NicheWithData>(), 12);
    assert_eq!(size_of::<Option<NicheWithData>>(), 12);
    assert_eq!(size_of::<Option<Option<NicheWithData>>>(), 12);
    assert_eq!(
        size_of::<Option<Option2<&(), Option<NicheWithData>>>>(),
        size_of::<(&(), NicheWithData)>()
    );

    pub enum FillPadding { A(std::num::NonZeroU8, u32), B }
    assert_eq!(size_of::<FillPadding>(), 8);
    assert_eq!(size_of::<Option<FillPadding>>(), 8);
    assert_eq!(size_of::<Option<Option<FillPadding>>>(), 8);

    assert_eq!(size_of::<Result<(std::num::NonZeroU8, u8, u8), u16>>(), 4);
    assert_eq!(size_of::<Option<Result<(std::num::NonZeroU8, u8, u8), u16>>>(), 4);
    assert_eq!(size_of::<Result<(std::num::NonZeroU8, u8, u8, u8), u16>>(), 4);

    assert_eq!(size_of::<EnumManyVariant<u16>>(), 6);
    assert_eq!(size_of::<EnumManyVariant<NicheU16>>(), 4);
    assert_eq!(size_of::<EnumManyVariant<Option<NicheU16>>>(), 4);
    assert_eq!(size_of::<EnumManyVariant<Option2<NicheU16,u8>>>(), 6);
    assert_eq!(size_of::<EnumManyVariant<Option<(NicheU16,u8)>>>(), 6);
}
