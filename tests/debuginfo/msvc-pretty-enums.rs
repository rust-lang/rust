//@ min-lldb-version: 1800
//@ ignore-gdb
//@ compile-flags:-g

// === LLDB TESTS ==================================================================================
// lldb-command:run

// lldb-command:v a
// lldb-check:(core::option::Option<msvc_pretty_enums::CStyleEnum>) a = { value = { 0 = Low } }

// lldb-command:v b
// lldb-check:(core::option::Option<msvc_pretty_enums::CStyleEnum>) b = { value = $discr$ = '\x01' }

// lldb-command:v c
// lldb-check:(msvc_pretty_enums::NicheLayoutEnum) c = { value = $discr$ = '\x11' }

// lldb-command:v d
// lldb-check:(msvc_pretty_enums::NicheLayoutEnum) d = { value = { my_data = High } }

// lldb-command:v e
// lldb-check:(msvc_pretty_enums::NicheLayoutEnum) e = { value = $discr$ = '\x13' }

// lldb-command:v h
// lldb-check:(core::option::Option<u32>) h = { value = { 0 = 12 } $discr$ = 1 }

// lldb-command:v i
// lldb-check:(core::option::Option<u32>) i = { value = $discr$ = 0 }

// lldb-command:v j
// lldb-check:(msvc_pretty_enums::CStyleEnum) j = High

// lldb-command:v k
// lldb-check:(core::option::Option<alloc::string::String>) k = { value = { 0 = "IAMA optional string!" { [0] = 'I' [1] = 'A' [2] = 'M' [3] = 'A' [4] = ' ' [5] = 'o' [6] = 'p' [7] = 't' [8] = 'i' [9] = 'o' [10] = 'n' [11] = 'a' [12] = 'l' [13] = ' ' [14] = 's' [15] = 't' [16] = 'r' [17] = 'i' [18] = 'n' [19] = 'g' [20] = '!' } } }

// lldb-command:v l
// lldb-check:(core::result::Result<u32, msvc_pretty_enums::Empty>) l = { value = { 0 = {} } }

// lldb-command:v niche128_some
// lldb-check:(core::option::Option<core::num::nonzero::NonZero<i128>>) niche128_some = { value = $discr$ = 123456 }

// lldb-command:v niche128_none
// lldb-check:(core::option::Option<core::num::nonzero::NonZero<i128>>) niche128_none = { value = $discr$ = 0 }

// lldb-command:v wrapping_niche128_untagged
// lldb-check:(msvc_pretty_enums::Wrapping128Niche) wrapping_niche128_untagged = { value = { 0 = { 0 = 340282366920938463463374607431768211454 } } }

// lldb-command:v wrapping_niche128_none1
// lldb-check:(msvc_pretty_enums::Wrapping128Niche) wrapping_niche128_none1 = { value = { 0 = { 0 = 2 } } }

// lldb-command:v direct_tag_128_a
// lldb-check:(msvc_pretty_enums::DirectTag128) direct_tag_128_a = { value = { 0 = 42 } $discr$ = 0 }

// lldb-command:v direct_tag_128_b
// lldb-check:(msvc_pretty_enums::DirectTag128) direct_tag_128_b = { value = { 0 = 137 } $discr$ = 1 }

// &u32 is incorrectly formatted and LLDB thinks it's a char* so skipping niche_w_fields_1_some

// lldb-command:v niche_w_fields_1_none
// lldb-check:(msvc_pretty_enums::NicheLayoutWithFields1) niche_w_fields_1_none = { value = { 0 = 99 } $discr$ = 1 }

// lldb-command:v niche_w_fields_2_some
// lldb-check:(msvc_pretty_enums::NicheLayoutWithFields2) niche_w_fields_2_some = { value = { 0 = 800 { __0 = { 0 = 800 } } 1 = 900 } $discr$ = 0 }

// lldb-command:v niche_w_fields_3_some
// lldb-check:(msvc_pretty_enums::NicheLayoutWithFields3) niche_w_fields_3_some = { value = { 0 = '\x89' 1 = true } }

// lldb-command:v niche_w_fields_3_niche3
// lldb-check:(msvc_pretty_enums::NicheLayoutWithFields3) niche_w_fields_3_niche3 = { value = { 0 = '"' } $discr$ = '\x04' }

// lldb-command:v arbitrary_discr1
// lldb-check:(msvc_pretty_enums::ArbitraryDiscr) arbitrary_discr1 = { value = { 0 = 1234 } $discr$ = 1000 }

// lldb-command:v arbitrary_discr2
// lldb-check:(msvc_pretty_enums::ArbitraryDiscr) arbitrary_discr2 = { value = { 0 = 5678 } $discr$ = 5000000 }

// === CDB TESTS ==================================================================================

// cdb-command: g
//
// cdb-command: dx a
// cdb-check:a                : Some [Type: enum2$<core::option::Option<msvc_pretty_enums::CStyleEnum> >]
// cdb-check:    [+0x000] __0              : Low (0x2) [Type: msvc_pretty_enums::CStyleEnum]
//
// cdb-command: dx b
// cdb-check:b                : None [Type: enum2$<core::option::Option<msvc_pretty_enums::CStyleEnum> >]
//
// cdb-command: dx c
// cdb-check:c                : Tag1 [Type: enum2$<msvc_pretty_enums::NicheLayoutEnum>]
//
// cdb-command: dx d
// cdb-check:d                : Data [Type: enum2$<msvc_pretty_enums::NicheLayoutEnum>]
// cdb-check:    [+0x000] my_data          : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]
//
// cdb-command: dx e
// cdb-check:e                : Tag2 [Type: enum2$<msvc_pretty_enums::NicheLayoutEnum>]
//
// cdb-command: dx f
// cdb-check:f                : Some [Type: enum2$<core::option::Option<ref$<u32> > >]
// cdb-check:    [+0x000] __0              : 0x[...] : 0x1 [Type: unsigned int *]
//
// cdb-command: dx g
// cdb-check:g                : None [Type: enum2$<core::option::Option<ref$<u32> > >]
//
// cdb-command: dx h
// cdb-check:h                : Some [Type: enum2$<core::option::Option<u32> >]
// cdb-check:    [+0x004] __0              : 0xc [Type: unsigned int]
//
// cdb-command: dx i
// cdb-check:i                : None [Type: enum2$<core::option::Option<u32> >]
//
// cdb-command: dx j
// cdb-check:j                : High (0x10) [Type: msvc_pretty_enums::CStyleEnum]
//
// cdb-command: dx k
// cdb-check:k                : Some [Type: enum2$<core::option::Option<alloc::string::String> >]
// cdb-check:    [+0x000] __0              : "IAMA optional string!" [Type: alloc::string::String]
//
// cdb-command: dx l
// cdb-check:l                : Ok [Type: enum2$<core::result::Result<u32,enum2$<msvc_pretty_enums::Empty> > >]
// cdb-check:    [+0x000] __0              : 0x2a [Type: unsigned int]
//
// cdb-command: dx niche128_some
// cdb-check: niche128_some    : Some [Type: enum2$<core::option::Option<core::num::nonzero::NonZero<i128> > >]
// Note: we can't actually read the value of the field because CDB cannot handle 128 bit integers.
// cdb-check:    [+0x000] __0 [...] [Type: core::num::nonzero::NonZero<i128>]
//
// cdb-command: dx niche128_none
// cdb-check: niche128_none    : None [Type: enum2$<core::option::Option<core::num::nonzero::NonZero<i128> > >]
//
// cdb-command: dx wrapping_niche128_untagged
// cdb-check: wrapping_niche128_untagged : X [Type: enum2$<msvc_pretty_enums::Wrapping128Niche>]
// cdb-check:    [+0x[...]] __0              [Type: msvc_pretty_enums::Wrapping128]
//
// cdb-command: dx wrapping_niche128_none1
// cdb-check: wrapping_niche128_none1 : Y [Type: enum2$<msvc_pretty_enums::Wrapping128Niche>]
// cdb-check:    [+0x[...]] __0              [Type: msvc_pretty_enums::Wrapping128]
//
// cdb-command: dx wrapping_niche128_none2
// cdb-check: wrapping_niche128_none2 : Z [Type: enum2$<msvc_pretty_enums::Wrapping128Niche>]
// cdb-check:    [+0x[...]] __0              [Type: msvc_pretty_enums::Wrapping128]
//
// cdb-command: dx direct_tag_128_a,d
// cdb-check: direct_tag_128_a,d : A [Type: enum2$<msvc_pretty_enums::DirectTag128>]
// cdb-check:     [+0x[...]] __0              : 42 [Type: unsigned int]
//
// cdb-command: dx direct_tag_128_b,d
// cdb-check: direct_tag_128_b,d : B [Type: enum2$<msvc_pretty_enums::DirectTag128>]
// cdb-check:     [+0x[...]] __0              : 137 [Type: unsigned int]
//
// cdb-command: dx niche_w_fields_1_some,d
// cdb-check: niche_w_fields_1_some,d : A [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields1>]
// cdb-check:     [+0x[...]] __0              : 0x[...] : 77 [Type: unsigned char *]
// cdb-check:     [+0x[...]] __1              : 7 [Type: unsigned int]
//
// cdb-command: dx niche_w_fields_1_none,d
// cdb-check: niche_w_fields_1_none,d : B [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields1>]
// cdb-check:     [+0x[...]] __0              : 99 [Type: unsigned int]
//
// cdb-command: dx niche_w_fields_2_some,d
// cdb-check: niche_w_fields_2_some,d : A [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields2>]
// cdb-check:     [+0x[...]] __0              : 800 [Type: core::num::nonzero::NonZero<u32>]
// cdb-check:     [+0x[...]] __1              : 900 [Type: unsigned __int64]
//
// cdb-command: dx niche_w_fields_2_none,d
// cdb-check: niche_w_fields_2_none,d : B [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields2>]
// cdb-check:     [+0x[...]] __0              : 1000 [Type: unsigned __int64]
//
// cdb-command: dx niche_w_fields_3_some,d
// cdb-check: niche_w_fields_3_some,d : A [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
// cdb-check:     [+0x[...]] __0              : 137 [Type: unsigned char]
// cdb-check:     [+0x[...]] __1              : true [Type: bool]
//
// cdb-command: dx niche_w_fields_3_niche1,d
// cdb-check: niche_w_fields_3_niche1,d : B [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
// cdb-check:     [+0x[...]] __0              : 12 [Type: unsigned char]
//
// cdb-command: dx niche_w_fields_3_niche2,d
// cdb-check: niche_w_fields_3_niche2,d : C [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
// cdb-check:     [+0x[...]] __0              : false [Type: bool]
//
// cdb-command: dx niche_w_fields_3_niche3,d
// cdb-check: niche_w_fields_3_niche3,d : D [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
// cdb-check:     [+0x[...]] __0              : 34 [Type: unsigned char]
//
// cdb-command: dx niche_w_fields_3_niche4,d
// cdb-check: niche_w_fields_3_niche4,d : E [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
// cdb-check:     [+0x[...]] __0              : 56 [Type: unsigned char]
//
// cdb-command: dx niche_w_fields_3_niche5,d
// cdb-check: niche_w_fields_3_niche5,d : F [Type: enum2$<msvc_pretty_enums::NicheLayoutWithFields3>]
//
// cdb-command: dx -r3 niche_w_fields_std_result_ok,d
// cdb-check: niche_w_fields_std_result_ok,d : Ok [Type: enum2$<core::result::Result<alloc::boxed::Box<slice2$<u8>,alloc::alloc::Global>,u64> >]
// cdb-check:    [+0x[...]] __0              [Type: alloc::boxed::Box<slice2$<u8>,alloc::alloc::Global>]
// cdb-check:        [+0x[...]] data_ptr         : [...]
// cdb-check:        [+0x[...]] length           : 3 [...]
//
// cdb-command: dx -r3 niche_w_fields_std_result_err,d
// cdb-check: niche_w_fields_std_result_err,d : Err [Type: enum2$<core::result::Result<alloc::boxed::Box<slice2$<u8>,alloc::alloc::Global>,u64> >]
// cdb-check:    [+0x[...]] __0              : 789 [Type: unsigned __int64]
//
// cdb-command: dx -r2 arbitrary_discr1,d
// cdb-check: arbitrary_discr1,d : Abc [Type: enum2$<msvc_pretty_enums::ArbitraryDiscr>]
// cdb-check:     [+0x[...]] __0              : 1234 [Type: unsigned int]
//
// cdb-command: dx -r2 arbitrary_discr2,d
// cdb-check: arbitrary_discr2,d : Def [Type: enum2$<msvc_pretty_enums::ArbitraryDiscr>]
// cdb-check:     [+0x[...]] __0              : 5678 [Type: unsigned int]
//
// cdb-command: dx c_style_u128_a
// cdb-check: c_style_u128_a   : A [Type: enum2$<msvc_pretty_enums::CStyleU128>]
//
// cdb-command: dx c_style_u128_b
// cdb-check: c_style_u128_b   : B [Type: enum2$<msvc_pretty_enums::CStyleU128>]
//
// cdb-command: dx c_style_u128_c
// cdb-check: c_style_u128_c   : C [Type: enum2$<msvc_pretty_enums::CStyleU128>]
//
// cdb-command: dx c_style_u128_d
// cdb-check: c_style_u128_d   : D [Type: enum2$<msvc_pretty_enums::CStyleU128>]
//
// cdb-command: dx c_style_i128_a
// cdb-check: c_style_i128_a   : A [Type: enum2$<msvc_pretty_enums::CStyleI128>]
//
// cdb-command: dx c_style_i128_b
// cdb-check: c_style_i128_b   : B [Type: enum2$<msvc_pretty_enums::CStyleI128>]
//
// cdb-command: dx c_style_i128_c
// cdb-check: c_style_i128_c   : C [Type: enum2$<msvc_pretty_enums::CStyleI128>]
//
// cdb-command: dx c_style_i128_d
// cdb-check: c_style_i128_d   : D [Type: enum2$<msvc_pretty_enums::CStyleI128>]
#![feature(rustc_attrs)]

use std::num::NonZero;

pub enum CStyleEnum {
    Low = 2,
    High = 16,
}

pub enum NicheLayoutEnum {
    Tag1,
    Data { my_data: CStyleEnum },
    Tag2,
}

pub enum Empty {}

// The following three types will use a niche layout once
// https://github.com/rust-lang/rust/pull/94075 is merged:
enum NicheLayoutWithFields1<'a> {
    A(&'a u8, u32),
    B(u32),
}

enum NicheLayoutWithFields2 {
    A(NonZero<u32>, u64),
    B(u64),
}

enum NicheLayoutWithFields3 {
    A(u8, bool),
    B(u8),
    C(bool),
    D(u8),
    E(u8),
    F,
}

#[rustc_layout_scalar_valid_range_start(340282366920938463463374607431768211454)]
#[rustc_layout_scalar_valid_range_end(1)]
#[repr(transparent)]
struct Wrapping128(u128);

// #[rustc_layout(debug)]
enum Wrapping128Niche {
    X(Wrapping128),
    Y,
    Z,
}

#[repr(i128)]
enum DirectTag128 {
    A(u32),
    B(u32),
}

#[repr(u32)]
enum ArbitraryDiscr {
    Abc(u32) = 1000,
    Def(u32) = 5000_000,
}

#[repr(u128)]
pub enum CStyleU128 {
    A = 0_u128,
    B = 1_u128,
    C = u64::MAX as u128 + 1,
    D = u128::MAX,
}

#[repr(i128)]
pub enum CStyleI128 {
    A = 0_i128,
    B = -1_i128,
    C = i128::MIN,
    D = i128::MAX,
}

fn main() {
    let a = Some(CStyleEnum::Low);
    let b = Option::<CStyleEnum>::None;
    let c = NicheLayoutEnum::Tag1;
    let d = NicheLayoutEnum::Data { my_data: CStyleEnum::High };
    let e = NicheLayoutEnum::Tag2;
    let f = Some(&1u32);
    let g = Option::<&'static u32>::None;
    let h = Some(12u32);
    let i = Option::<u32>::None;
    let j = CStyleEnum::High;
    let k = Some("IAMA optional string!".to_string());
    let l = Result::<u32, Empty>::Ok(42);
    let niche128_some = NonZero::new(123456i128);
    let niche128_none: Option<NonZero<i128>> = None;

    let wrapping_niche128_untagged =
        unsafe { Wrapping128Niche::X(Wrapping128(340282366920938463463374607431768211454)) };
    let wrapping_niche128_none1 = Wrapping128Niche::Y;
    let wrapping_niche128_none2 = Wrapping128Niche::Z;

    let direct_tag_128_a = DirectTag128::A(42);
    let direct_tag_128_b = DirectTag128::B(137);

    let niche_w_fields_1_some = NicheLayoutWithFields1::A(&77, 7);
    let niche_w_fields_1_none = NicheLayoutWithFields1::B(99);

    let niche_w_fields_2_some = NicheLayoutWithFields2::A(NonZero::new(800).unwrap(), 900);
    let niche_w_fields_2_none = NicheLayoutWithFields2::B(1000);

    let niche_w_fields_3_some = NicheLayoutWithFields3::A(137, true);
    let niche_w_fields_3_niche1 = NicheLayoutWithFields3::B(12);
    let niche_w_fields_3_niche2 = NicheLayoutWithFields3::C(false);
    let niche_w_fields_3_niche3 = NicheLayoutWithFields3::D(34);
    let niche_w_fields_3_niche4 = NicheLayoutWithFields3::E(56);
    let niche_w_fields_3_niche5 = NicheLayoutWithFields3::F;

    let niche_w_fields_std_result_ok: Result<Box<[u8]>, u64> = Ok(vec![1, 2, 3].into());
    let niche_w_fields_std_result_err: Result<Box<[u8]>, u64> = Err(789);

    let arbitrary_discr1 = ArbitraryDiscr::Abc(1234);
    let arbitrary_discr2 = ArbitraryDiscr::Def(5678);

    let c_style_u128_a = CStyleU128::A;
    let c_style_u128_b = CStyleU128::B;
    let c_style_u128_c = CStyleU128::C;
    let c_style_u128_d = CStyleU128::D;

    let c_style_i128_a = CStyleI128::A;
    let c_style_i128_b = CStyleI128::B;
    let c_style_i128_c = CStyleI128::C;
    let c_style_i128_d = CStyleI128::D;

    zzz(); // #break
}

fn zzz() {
    ()
}
