// run-pass

// This file tests repr(no_niche), which causes an struct/enum to hide
// any niche space that may exist in its internal state from the
// context it appears in.

// Here are the axes this test is seeking to cover:
//
// repr annotation:
// visible: (); cloaked: (no_niche); transparent: (transparent); shadowy: (transparent, no_niche)
//
// enum vs struct
//
// niche-type via type-parameter vs inline declaration

#![feature(decl_macro)]
#![feature(no_niche)]

use std::mem::size_of;
use std::num::{NonZeroU8, NonZeroU16};

mod struct_inline {
    use std::num::NonZeroU16 as N16;

    #[derive(Debug)] pub struct Visible(N16);

    #[repr(no_niche)]
    #[derive(Debug)] pub struct Cloaked(N16);

    #[repr(transparent)]
    #[derive(Debug)] pub struct Transparent(N16);

    #[repr(transparent, no_niche)]
    #[derive(Debug)] pub struct Shadowy(N16);
}

mod struct_param {
    #[derive(Debug)] pub struct Visible<T>(T);

    #[repr(no_niche)]
    #[derive(Debug)] pub struct Cloaked<T>(T);

    #[repr(transparent)]
    #[derive(Debug)] pub struct Transparent<T>(T);

    #[repr(transparent, no_niche)]
    #[derive(Debug)] pub struct Shadowy<T>(T);
}

mod enum_inline {
    use crate::two_fifty_six_variant_enum;
    use std::num::{NonZeroU8 as N8, NonZeroU16 as N16};

    #[derive(Debug)] pub enum Visible1 { _A(N16), }

    #[repr(no_niche)]
    #[derive(Debug)] pub enum Cloaked1 { _A(N16), }

    // (N.B.: transparent enums must be univariant)
    #[repr(transparent)]
    #[derive(Debug)] pub enum Transparent { _A(N16), }

    #[repr(transparent, no_niche)]
    #[derive(Debug)] pub enum Shadowy { _A(N16), }

    // including multivariant enums for completeness. Payload and
    // number of variants (i.e. discriminant size) have been chosen so
    // that layout including discriminant is 4 bytes, with no space in
    // padding to hide another discrimnant from the surrounding
    // context.
    //
    // (Note that multivariant enums cannot usefully expose a niche in
    // general; this test is relying on that.)
    two_fifty_six_variant_enum!(Visible2, N8);

    #[repr(no_niche)]
    two_fifty_six_variant_enum!(Cloaked2, N8);
}

mod enum_param {
    use super::two_fifty_six_variant_enum;

    #[derive(Debug)] pub enum Visible1<T> { _A(T), }

    #[repr(no_niche)]
    #[derive(Debug)] pub enum Cloaked1<T> { _A(T), }

    // (N.B.: transparent enums must be univariant)
    #[repr(transparent)]
    #[derive(Debug)] pub enum Transparent<T> { _A(T), }

    #[repr(transparent, no_niche)]
    #[derive(Debug)] pub enum Shadowy<T> { _A(T), }

    // including multivariant enums for completeness. Same notes apply
    // here as above (assuming `T` is instantiated with `NonZeroU8`).
    two_fifty_six_variant_enum!(Visible2<T>);

    #[repr(no_niche)]
    two_fifty_six_variant_enum!(Cloaked2<T>);
}

fn main() {
    // sanity-checks
    assert_eq!(size_of::<struct_inline::Visible>(),               2);
    assert_eq!(size_of::<struct_inline::Cloaked>(),               2);
    assert_eq!(size_of::<struct_inline::Transparent>(),           2);
    assert_eq!(size_of::<struct_inline::Shadowy>(),               2);

    assert_eq!(size_of::<struct_param::Visible<NonZeroU16>>(), 2);
    assert_eq!(size_of::<struct_param::Cloaked<NonZeroU16>>(), 2);
    assert_eq!(size_of::<struct_param::Transparent<NonZeroU16>>(), 2);
    assert_eq!(size_of::<struct_param::Shadowy<NonZeroU16>>(), 2);

    assert_eq!(size_of::<enum_inline::Visible1>(),    2);
    assert_eq!(size_of::<enum_inline::Cloaked1>(),    2);
    assert_eq!(size_of::<enum_inline::Transparent>(), 2); // transparent enums are univariant
    assert_eq!(size_of::<enum_inline::Shadowy>(),     2);
    assert_eq!(size_of::<enum_inline::Visible2>(),    4);
    assert_eq!(size_of::<enum_inline::Cloaked2>(),    4);

    assert_eq!(size_of::<enum_param::Visible1<NonZeroU16>>(),    2);
    assert_eq!(size_of::<enum_param::Cloaked1<NonZeroU16>>(),    2);
    assert_eq!(size_of::<enum_param::Transparent<NonZeroU16>>(), 2);
    assert_eq!(size_of::<enum_param::Shadowy<NonZeroU16>>(),     2);
    assert_eq!(size_of::<enum_param::Visible2<NonZeroU8>>(),     4);
    assert_eq!(size_of::<enum_param::Cloaked2<NonZeroU8>>(),     4);

    // now the actual tests of no_niche: how do inputs above compose
    // with `Option` type constructor. The cases with a `_+2` are the
    // ones where no_niche fires.
    assert_eq!(size_of::<Option<struct_inline::Visible>>(),       2);
    assert_eq!(size_of::<Option<struct_inline::Cloaked>>(),       2+2);
    assert_eq!(size_of::<Option<struct_inline::Transparent>>(),   2);
    assert_eq!(size_of::<Option<struct_inline::Shadowy>>(),       2+2);

    assert_eq!(size_of::<Option<struct_param::Visible<NonZeroU16>>>(),     2);
    assert_eq!(size_of::<Option<struct_param::Cloaked<NonZeroU16>>>(),     2+2);
    assert_eq!(size_of::<Option<struct_param::Transparent<NonZeroU16>>>(), 2);
    assert_eq!(size_of::<Option<struct_param::Shadowy<NonZeroU16>>>(),     2+2);

    assert_eq!(size_of::<Option<enum_inline::Visible1>>(),    2);
    assert_eq!(size_of::<Option<enum_inline::Cloaked1>>(),    2+2);
    assert_eq!(size_of::<Option<enum_inline::Transparent>>(), 2);
    assert_eq!(size_of::<Option<enum_inline::Shadowy>>(),     2+2);
    // cannot use niche of multivariant payload
    assert_eq!(size_of::<Option<enum_inline::Visible2>>(),    4+2);
    assert_eq!(size_of::<Option<enum_inline::Cloaked2>>(),    4+2);

    assert_eq!(size_of::<Option<enum_param::Visible1<NonZeroU16>>>(),    2);
    assert_eq!(size_of::<Option<enum_param::Cloaked1<NonZeroU16>>>(),    2+2);
    assert_eq!(size_of::<Option<enum_param::Transparent<NonZeroU16>>>(), 2);
    assert_eq!(size_of::<Option<enum_param::Shadowy<NonZeroU16>>>(),     2+2);
    // cannot use niche of multivariant payload
    assert_eq!(size_of::<Option<enum_param::Visible2<NonZeroU8>>>(),    4+2);
    assert_eq!(size_of::<Option<enum_param::Cloaked2<NonZeroU8>>>(),    4+2);
}

macro two_fifty_six_variant_enum {
    ($name:ident<$param:ident>) => {
        #[derive(Debug)]
        pub enum $name<$param> {
            _V00($param, u16), _V01(u16, $param), _V02($param, u16), _V03(u16, $param),
            _V04($param, u16), _V05(u16, $param), _V06($param, u16), _V07(u16, $param),
            _V08($param, u16), _V09(u16, $param), _V0a($param, u16), _V0b(u16, $param),
            _V0c($param, u16), _V0d(u16, $param), _V0e($param, u16), _V0f(u16, $param),

            _V10($param, u16), _V11(u16, $param), _V12($param, u16), _V13(u16, $param),
            _V14($param, u16), _V15(u16, $param), _V16($param, u16), _V17(u16, $param),
            _V18($param, u16), _V19(u16, $param), _V1a($param, u16), _V1b(u16, $param),
            _V1c($param, u16), _V1d(u16, $param), _V1e($param, u16), _V1f(u16, $param),

            _V20($param, u16), _V21(u16, $param), _V22($param, u16), _V23(u16, $param),
            _V24($param, u16), _V25(u16, $param), _V26($param, u16), _V27(u16, $param),
            _V28($param, u16), _V29(u16, $param), _V2a($param, u16), _V2b(u16, $param),
            _V2c($param, u16), _V2d(u16, $param), _V2e($param, u16), _V2f(u16, $param),

            _V30($param, u16), _V31(u16, $param), _V32($param, u16), _V33(u16, $param),
            _V34($param, u16), _V35(u16, $param), _V36($param, u16), _V37(u16, $param),
            _V38($param, u16), _V39(u16, $param), _V3a($param, u16), _V3b(u16, $param),
            _V3c($param, u16), _V3d(u16, $param), _V3e($param, u16), _V3f(u16, $param),

            _V40($param, u16), _V41(u16, $param), _V42($param, u16), _V43(u16, $param),
            _V44($param, u16), _V45(u16, $param), _V46($param, u16), _V47(u16, $param),
            _V48($param, u16), _V49(u16, $param), _V4a($param, u16), _V4b(u16, $param),
            _V4c($param, u16), _V4d(u16, $param), _V4e($param, u16), _V4f(u16, $param),

            _V50($param, u16), _V51(u16, $param), _V52($param, u16), _V53(u16, $param),
            _V54($param, u16), _V55(u16, $param), _V56($param, u16), _V57(u16, $param),
            _V58($param, u16), _V59(u16, $param), _V5a($param, u16), _V5b(u16, $param),
            _V5c($param, u16), _V5d(u16, $param), _V5e($param, u16), _V5f(u16, $param),

            _V60($param, u16), _V61(u16, $param), _V62($param, u16), _V63(u16, $param),
            _V64($param, u16), _V65(u16, $param), _V66($param, u16), _V67(u16, $param),
            _V68($param, u16), _V69(u16, $param), _V6a($param, u16), _V6b(u16, $param),
            _V6c($param, u16), _V6d(u16, $param), _V6e($param, u16), _V6f(u16, $param),

            _V70($param, u16), _V71(u16, $param), _V72($param, u16), _V73(u16, $param),
            _V74($param, u16), _V75(u16, $param), _V76($param, u16), _V77(u16, $param),
            _V78($param, u16), _V79(u16, $param), _V7a($param, u16), _V7b(u16, $param),
            _V7c($param, u16), _V7d(u16, $param), _V7e($param, u16), _V7f(u16, $param),

            _V80($param, u16), _V81(u16, $param), _V82($param, u16), _V83(u16, $param),
            _V84($param, u16), _V85(u16, $param), _V86($param, u16), _V87(u16, $param),
            _V88($param, u16), _V89(u16, $param), _V8a($param, u16), _V8b(u16, $param),
            _V8c($param, u16), _V8d(u16, $param), _V8e($param, u16), _V8f(u16, $param),

            _V90($param, u16), _V91(u16, $param), _V92($param, u16), _V93(u16, $param),
            _V94($param, u16), _V95(u16, $param), _V96($param, u16), _V97(u16, $param),
            _V98($param, u16), _V99(u16, $param), _V9a($param, u16), _V9b(u16, $param),
            _V9c($param, u16), _V9d(u16, $param), _V9e($param, u16), _V9f(u16, $param),

            _Va0($param, u16), _Va1(u16, $param), _Va2($param, u16), _Va3(u16, $param),
            _Va4($param, u16), _Va5(u16, $param), _Va6($param, u16), _Va7(u16, $param),
            _Va8($param, u16), _Va9(u16, $param), _Vaa($param, u16), _Vab(u16, $param),
            _Vac($param, u16), _Vad(u16, $param), _Vae($param, u16), _Vaf(u16, $param),

            _Vb0($param, u16), _Vb1(u16, $param), _Vb2($param, u16), _Vb3(u16, $param),
            _Vb4($param, u16), _Vb5(u16, $param), _Vb6($param, u16), _Vb7(u16, $param),
            _Vb8($param, u16), _Vb9(u16, $param), _Vba($param, u16), _Vbb(u16, $param),
            _Vbc($param, u16), _Vbd(u16, $param), _Vbe($param, u16), _Vbf(u16, $param),

            _Vc0($param, u16), _Vc1(u16, $param), _Vc2($param, u16), _Vc3(u16, $param),
            _Vc4($param, u16), _Vc5(u16, $param), _Vc6($param, u16), _Vc7(u16, $param),
            _Vc8($param, u16), _Vc9(u16, $param), _Vca($param, u16), _Vcb(u16, $param),
            _Vcc($param, u16), _Vcd(u16, $param), _Vce($param, u16), _Vcf(u16, $param),

            _Vd0($param, u16), _Vd1(u16, $param), _Vd2($param, u16), _Vd3(u16, $param),
            _Vd4($param, u16), _Vd5(u16, $param), _Vd6($param, u16), _Vd7(u16, $param),
            _Vd8($param, u16), _Vd9(u16, $param), _Vda($param, u16), _Vdb(u16, $param),
            _Vdc($param, u16), _Vdd(u16, $param), _Vde($param, u16), _Vdf(u16, $param),

            _Ve0($param, u16), _Ve1(u16, $param), _Ve2($param, u16), _Ve3(u16, $param),
            _Ve4($param, u16), _Ve5(u16, $param), _Ve6($param, u16), _Ve7(u16, $param),
            _Ve8($param, u16), _Ve9(u16, $param), _Vea($param, u16), _Veb(u16, $param),
            _Vec($param, u16), _Ved(u16, $param), _Vee($param, u16), _Vef(u16, $param),

            _Vf0($param, u16), _Vf1(u16, $param), _Vf2($param, u16), _Vf3(u16, $param),
            _Vf4($param, u16), _Vf5(u16, $param), _Vf6($param, u16), _Vf7(u16, $param),
            _Vf8($param, u16), _Vf9(u16, $param), _Vfa($param, u16), _Vfb(u16, $param),
            _Vfc($param, u16), _Vfd(u16, $param), _Vfe($param, u16), _Vff(u16, $param),
        }
    },

    ($name:ident, $param:ty) => {
        #[derive(Debug)]
        pub enum $name {
            _V00($param, u16), _V01(u16, $param), _V02($param, u16), _V03(u16, $param),
            _V04($param, u16), _V05(u16, $param), _V06($param, u16), _V07(u16, $param),
            _V08($param, u16), _V09(u16, $param), _V0a($param, u16), _V0b(u16, $param),
            _V0c($param, u16), _V0d(u16, $param), _V0e($param, u16), _V0f(u16, $param),

            _V10($param, u16), _V11(u16, $param), _V12($param, u16), _V13(u16, $param),
            _V14($param, u16), _V15(u16, $param), _V16($param, u16), _V17(u16, $param),
            _V18($param, u16), _V19(u16, $param), _V1a($param, u16), _V1b(u16, $param),
            _V1c($param, u16), _V1d(u16, $param), _V1e($param, u16), _V1f(u16, $param),

            _V20($param, u16), _V21(u16, $param), _V22($param, u16), _V23(u16, $param),
            _V24($param, u16), _V25(u16, $param), _V26($param, u16), _V27(u16, $param),
            _V28($param, u16), _V29(u16, $param), _V2a($param, u16), _V2b(u16, $param),
            _V2c($param, u16), _V2d(u16, $param), _V2e($param, u16), _V2f(u16, $param),

            _V30($param, u16), _V31(u16, $param), _V32($param, u16), _V33(u16, $param),
            _V34($param, u16), _V35(u16, $param), _V36($param, u16), _V37(u16, $param),
            _V38($param, u16), _V39(u16, $param), _V3a($param, u16), _V3b(u16, $param),
            _V3c($param, u16), _V3d(u16, $param), _V3e($param, u16), _V3f(u16, $param),

            _V40($param, u16), _V41(u16, $param), _V42($param, u16), _V43(u16, $param),
            _V44($param, u16), _V45(u16, $param), _V46($param, u16), _V47(u16, $param),
            _V48($param, u16), _V49(u16, $param), _V4a($param, u16), _V4b(u16, $param),
            _V4c($param, u16), _V4d(u16, $param), _V4e($param, u16), _V4f(u16, $param),

            _V50($param, u16), _V51(u16, $param), _V52($param, u16), _V53(u16, $param),
            _V54($param, u16), _V55(u16, $param), _V56($param, u16), _V57(u16, $param),
            _V58($param, u16), _V59(u16, $param), _V5a($param, u16), _V5b(u16, $param),
            _V5c($param, u16), _V5d(u16, $param), _V5e($param, u16), _V5f(u16, $param),

            _V60($param, u16), _V61(u16, $param), _V62($param, u16), _V63(u16, $param),
            _V64($param, u16), _V65(u16, $param), _V66($param, u16), _V67(u16, $param),
            _V68($param, u16), _V69(u16, $param), _V6a($param, u16), _V6b(u16, $param),
            _V6c($param, u16), _V6d(u16, $param), _V6e($param, u16), _V6f(u16, $param),

            _V70($param, u16), _V71(u16, $param), _V72($param, u16), _V73(u16, $param),
            _V74($param, u16), _V75(u16, $param), _V76($param, u16), _V77(u16, $param),
            _V78($param, u16), _V79(u16, $param), _V7a($param, u16), _V7b(u16, $param),
            _V7c($param, u16), _V7d(u16, $param), _V7e($param, u16), _V7f(u16, $param),

            _V80($param, u16), _V81(u16, $param), _V82($param, u16), _V83(u16, $param),
            _V84($param, u16), _V85(u16, $param), _V86($param, u16), _V87(u16, $param),
            _V88($param, u16), _V89(u16, $param), _V8a($param, u16), _V8b(u16, $param),
            _V8c($param, u16), _V8d(u16, $param), _V8e($param, u16), _V8f(u16, $param),

            _V90($param, u16), _V91(u16, $param), _V92($param, u16), _V93(u16, $param),
            _V94($param, u16), _V95(u16, $param), _V96($param, u16), _V97(u16, $param),
            _V98($param, u16), _V99(u16, $param), _V9a($param, u16), _V9b(u16, $param),
            _V9c($param, u16), _V9d(u16, $param), _V9e($param, u16), _V9f(u16, $param),

            _Va0($param, u16), _Va1(u16, $param), _Va2($param, u16), _Va3(u16, $param),
            _Va4($param, u16), _Va5(u16, $param), _Va6($param, u16), _Va7(u16, $param),
            _Va8($param, u16), _Va9(u16, $param), _Vaa($param, u16), _Vab(u16, $param),
            _Vac($param, u16), _Vad(u16, $param), _Vae($param, u16), _Vaf(u16, $param),

            _Vb0($param, u16), _Vb1(u16, $param), _Vb2($param, u16), _Vb3(u16, $param),
            _Vb4($param, u16), _Vb5(u16, $param), _Vb6($param, u16), _Vb7(u16, $param),
            _Vb8($param, u16), _Vb9(u16, $param), _Vba($param, u16), _Vbb(u16, $param),
            _Vbc($param, u16), _Vbd(u16, $param), _Vbe($param, u16), _Vbf(u16, $param),

            _Vc0($param, u16), _Vc1(u16, $param), _Vc2($param, u16), _Vc3(u16, $param),
            _Vc4($param, u16), _Vc5(u16, $param), _Vc6($param, u16), _Vc7(u16, $param),
            _Vc8($param, u16), _Vc9(u16, $param), _Vca($param, u16), _Vcb(u16, $param),
            _Vcc($param, u16), _Vcd(u16, $param), _Vce($param, u16), _Vcf(u16, $param),

            _Vd0($param, u16), _Vd1(u16, $param), _Vd2($param, u16), _Vd3(u16, $param),
            _Vd4($param, u16), _Vd5(u16, $param), _Vd6($param, u16), _Vd7(u16, $param),
            _Vd8($param, u16), _Vd9(u16, $param), _Vda($param, u16), _Vdb(u16, $param),
            _Vdc($param, u16), _Vdd(u16, $param), _Vde($param, u16), _Vdf(u16, $param),

            _Ve0($param, u16), _Ve1(u16, $param), _Ve2($param, u16), _Ve3(u16, $param),
            _Ve4($param, u16), _Ve5(u16, $param), _Ve6($param, u16), _Ve7(u16, $param),
            _Ve8($param, u16), _Ve9(u16, $param), _Vea($param, u16), _Veb(u16, $param),
            _Vec($param, u16), _Ved(u16, $param), _Vee($param, u16), _Vef(u16, $param),

            _Vf0($param, u16), _Vf1(u16, $param), _Vf2($param, u16), _Vf3(u16, $param),
            _Vf4($param, u16), _Vf5(u16, $param), _Vf6($param, u16), _Vf7(u16, $param),
            _Vf8($param, u16), _Vf9(u16, $param), _Vfa($param, u16), _Vfb(u16, $param),
            _Vfc($param, u16), _Vfd(u16, $param), _Vfe($param, u16), _Vff(u16, $param),
        }
    }
}
