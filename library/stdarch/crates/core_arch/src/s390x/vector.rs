//! s390x vector intrinsics.
//!
//! For more info see the [Reference Summary] or the online [IBM docs].
//!
//! [Reference Summary]: https://www.ibm.com/support/pages/sites/default/files/2021-05/SA22-7871-10.pdf
//! [IBM docs]: https://www.ibm.com/docs/en/zos/2.4.0?topic=support-vector-built-in-functions

#![allow(non_camel_case_types)]

use crate::{core_arch::simd::*, intrinsics::simd::*, mem::MaybeUninit, mem::transmute};

#[cfg(test)]
use stdarch_test::assert_instr;

use super::macros::*;

types! {
    #![unstable(feature = "stdarch_s390x", issue = "135681")]

    /// s390x-specific 128-bit wide vector of sixteen packed `i8`
    pub struct vector_signed_char(16 x i8);
    /// s390x-specific 128-bit wide vector of sixteen packed `u8`
    pub struct vector_unsigned_char(16 x u8);
    /// s390x-specific 128-bit wide vector mask of sixteen packed elements
    pub struct vector_bool_char(16 x i8);

    /// s390x-specific 128-bit wide vector of eight packed `i16`
    pub struct vector_signed_short(8 x i16);
    /// s390x-specific 128-bit wide vector of eight packed `u16`
    pub struct vector_unsigned_short(8 x u16);
    /// s390x-specific 128-bit wide vector mask of eight packed elements
    pub struct vector_bool_short(8 x i16);

    /// s390x-specific 128-bit wide vector of four packed `i32`
    pub struct vector_signed_int(4 x i32);
    /// s390x-specific 128-bit wide vector of four packed `u32`
    pub struct vector_unsigned_int(4 x u32);
    /// s390x-specific 128-bit wide vector mask of four packed elements
    pub struct vector_bool_int(4 x i32);

    /// s390x-specific 128-bit wide vector of two packed `i64`
    pub struct vector_signed_long_long(2 x i64);
    /// s390x-specific 128-bit wide vector of two packed `u64`
    pub struct vector_unsigned_long_long(2 x u64);
    /// s390x-specific 128-bit wide vector mask of two packed elements
    pub struct vector_bool_long_long(2 x i64);

    /// s390x-specific 128-bit wide vector of four packed `f32`
    pub struct vector_float(4 x f32);
    /// s390x-specific 128-bit wide vector of two packed `f64`
    pub struct vector_double(2 x f64);
}

#[repr(C, packed)]
struct PackedTuple<T, U> {
    x: T,
    y: U,
}

#[allow(improper_ctypes)]
#[rustfmt::skip]
unsafe extern "unadjusted" {
    #[link_name = "llvm.smax.v16i8"] fn vmxb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.smax.v8i16"] fn vmxh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.smax.v4i32"] fn vmxf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.smax.v2i64"] fn vmxg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.umax.v16i8"] fn vmxlb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.umax.v8i16"] fn vmxlh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.umax.v4i32"] fn vmxlf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.umax.v2i64"] fn vmxlg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.smin.v16i8"] fn vmnb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.smin.v8i16"] fn vmnh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.smin.v4i32"] fn vmnf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.smin.v2i64"] fn vmng(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.umin.v16i8"] fn vmnlb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.umin.v8i16"] fn vmnlh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.umin.v4i32"] fn vmnlf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.umin.v2i64"] fn vmnlg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.nearbyint.v4f32"] fn nearbyint_v4f32(a: vector_float) -> vector_float;
    #[link_name = "llvm.nearbyint.v2f64"] fn nearbyint_v2f64(a: vector_double) -> vector_double;

    #[link_name = "llvm.roundeven.v4f32"] fn roundeven_v4f32(a: vector_float) -> vector_float;
    #[link_name = "llvm.roundeven.v2f64"] fn roundeven_v2f64(a: vector_double) -> vector_double;

    #[link_name = "llvm.s390.vsra"] fn vsra(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsrl"] fn vsrl(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsl"] fn vsl(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;

    #[link_name = "llvm.s390.vsrab"] fn vsrab(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vsrlb"] fn vsrlb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vslb"] fn vslb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;

    #[link_name = "llvm.s390.vsrd"] fn vsrd(a: i8x16, b: i8x16, c: u32) -> i8x16;

    #[link_name = "llvm.s390.verimb"] fn verimb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char, d: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.verimh"] fn verimh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short, d: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.verimf"] fn verimf(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int, d: i32) -> vector_signed_int;
    #[link_name = "llvm.s390.verimg"] fn verimg(a: vector_signed_long_long, b: vector_signed_long_long, c: vector_signed_long_long, d: i32) -> vector_signed_long_long;

    #[link_name = "llvm.s390.vperm"] fn vperm(a: vector_signed_char, b: vector_signed_char, c: vector_unsigned_char) -> vector_signed_char;

    #[link_name = "llvm.s390.vsumb"] fn vsumb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vsumh"] fn vsumh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vsumgh"] fn vsumgh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_long_long;
    #[link_name = "llvm.s390.vsumgf"] fn vsumgf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vsumqf"] fn vsumqf(a: vector_unsigned_int, b: vector_unsigned_int) -> u128;
    #[link_name = "llvm.s390.vsumqg"] fn vsumqg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> u128;

    #[link_name = "llvm.s390.vaccq"] fn vaccq(a: u128, b: u128) -> u128;
    #[link_name = "llvm.s390.vacccq"] fn vacccq(a: u128, b: u128, c: u128) -> u128;

    #[link_name = "llvm.s390.vscbiq"] fn vscbiq(a: u128, b: u128) -> u128;
    #[link_name = "llvm.s390.vsbiq"] fn vsbiq(a: u128, b: u128, c: u128) -> u128;
    #[link_name = "llvm.s390.vsbcbiq"] fn vsbcbiq(a: u128, b: u128, c: u128) -> u128;

    #[link_name = "llvm.s390.vacq"] fn vacq(a: u128, b: u128, c: u128) -> u128;

    #[link_name = "llvm.s390.vscbib"] fn vscbib(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vscbih"] fn vscbih(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vscbif"] fn vscbif(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vscbig"] fn vscbig(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vfaeb"] fn vfaeb(a: vector_signed_char, b: vector_signed_char, c: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.vfaeh"] fn vfaeh(a: vector_signed_short, b: vector_signed_short, c: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.vfaef"] fn vfaef(a: vector_signed_int, b: vector_signed_int, c: i32) -> vector_signed_int;

    #[link_name = "llvm.s390.vfaezb"] fn vfaezb(a: vector_signed_char, b: vector_signed_char, c: i32) -> vector_signed_char;
    #[link_name = "llvm.s390.vfaezh"] fn vfaezh(a: vector_signed_short, b: vector_signed_short, c: i32) -> vector_signed_short;
    #[link_name = "llvm.s390.vfaezf"] fn vfaezf(a: vector_signed_int, b: vector_signed_int, c: i32) -> vector_signed_int;

    #[link_name = "llvm.s390.vfaebs"] fn vfaebs(a: vector_signed_char, b: vector_signed_char, c: i32) -> PackedTuple<vector_signed_char, i32>;
    #[link_name = "llvm.s390.vfaehs"] fn vfaehs(a: vector_signed_short, b: vector_signed_short, c: i32) -> PackedTuple<vector_signed_short, i32>;
    #[link_name = "llvm.s390.vfaefs"] fn vfaefs(a: vector_signed_int, b: vector_signed_int, c: i32) -> PackedTuple<vector_signed_int, i32>;

    #[link_name = "llvm.s390.vfaezbs"] fn vfaezbs(a: vector_signed_char, b: vector_signed_char, c: i32) -> PackedTuple<vector_signed_char, i32>;
    #[link_name = "llvm.s390.vfaezhs"] fn vfaezhs(a: vector_signed_short, b: vector_signed_short, c: i32) -> PackedTuple<vector_signed_short, i32>;
    #[link_name = "llvm.s390.vfaezfs"] fn vfaezfs(a: vector_signed_int, b: vector_signed_int, c: i32) -> PackedTuple<vector_signed_int, i32>;

    #[link_name = "llvm.s390.vll"] fn vll(a: u32, b: *const u8) -> vector_signed_char;
    #[link_name = "llvm.s390.vstl"] fn vstl(a: vector_signed_char, b: u32, c: *mut u8);

    #[link_name = "llvm.s390.vlrl"] fn vlrl(a: u32, b: *const u8) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vstrl"] fn vstrl(a: vector_unsigned_char, b: u32, c: *mut u8);

    #[link_name = "llvm.s390.lcbb"] fn lcbb(a: *const u8, b: u32) -> u32;
    #[link_name = "llvm.s390.vlbb"] fn vlbb(a: *const u8, b: u32) -> MaybeUninit<vector_signed_char>;

    #[link_name = "llvm.s390.vpksh"] fn vpksh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_char;
    #[link_name = "llvm.s390.vpksf"] fn vpksf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_short;
    #[link_name = "llvm.s390.vpksg"] fn vpksg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_int;

    #[link_name = "llvm.s390.vpklsh"] fn vpklsh(a: vector_signed_short, b: vector_signed_short) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vpklsf"] fn vpklsf(a: vector_signed_int, b: vector_signed_int) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vpklsg"] fn vpklsg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vpkshs"] fn vpkshs(a: vector_signed_short, b: vector_signed_short) -> PackedTuple<vector_signed_char, i32>;
    #[link_name = "llvm.s390.vpksfs"] fn vpksfs(a: vector_signed_int, b: vector_signed_int) -> PackedTuple<vector_signed_short, i32>;
    #[link_name = "llvm.s390.vpksgs"] fn vpksgs(a: vector_signed_long_long, b: vector_signed_long_long) -> PackedTuple<vector_signed_int, i32>;

    #[link_name = "llvm.s390.vpklshs"] fn vpklshs(a: vector_unsigned_short, b: vector_unsigned_short) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vpklsfs"] fn vpklsfs(a: vector_unsigned_int, b: vector_unsigned_int) -> PackedTuple<vector_unsigned_short, i32>;
    #[link_name = "llvm.s390.vpklsgs"] fn vpklsgs(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> PackedTuple<vector_unsigned_int, i32>;

    #[link_name = "llvm.s390.vavgb"] fn vavgb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vavgh"] fn vavgh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vavgf"] fn vavgf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.s390.vavgg"] fn vavgg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.s390.vavglb"] fn vavglb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vavglh"] fn vavglh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vavglf"] fn vavglf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vavglg"] fn vavglg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vcksm"] fn vcksm(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vmhb"] fn vmhb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vmhh"] fn vmhh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vmhf"] fn vmhf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.s390.vmlhb"] fn vmlhb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vmlhh"] fn vmlhh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vmlhf"] fn vmlhf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vmaeb"] fn vmaeb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vmaeh"] fn vmaeh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.s390.vmaef"] fn vmaef(a: vector_signed_int, b: vector_signed_int, c: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.s390.vmaleb"] fn vmaleb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vmaleh"] fn vmaleh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vmalef"] fn vmalef(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vmaob"] fn vmaob(a: vector_signed_char, b: vector_signed_char, c: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vmaoh"] fn vmaoh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_int) -> vector_signed_int;
    #[link_name = "llvm.s390.vmaof"] fn vmaof(a: vector_signed_int, b: vector_signed_int, c: vector_signed_long_long) -> vector_signed_long_long;

    #[link_name = "llvm.s390.vmalob"] fn vmalob(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vmaloh"] fn vmaloh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vmalof"] fn vmalof(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vmahb"] fn vmahb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vmahh"] fn vmahh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vmahf"] fn vmahf(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.s390.vmalhb"] fn vmalhb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vmalhh"] fn vmalhh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vmalhf"] fn vmalhf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vmalb"] fn vmalb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char) -> vector_signed_char;
    #[link_name = "llvm.s390.vmalh"] fn vmalh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short) -> vector_signed_short;
    #[link_name = "llvm.s390.vmalf"] fn vmalf(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int) -> vector_signed_int;

    #[link_name = "llvm.s390.vmallb"] fn vmallb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vmallh"] fn vmallh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vmallf"] fn vmallf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vgfmb"] fn vgfmb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vgfmh"] fn vgfmh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vgfmf"] fn vgfmf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long;
    #[link_name = "llvm.s390.vgfmg"] fn vgfmg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> u128;

    #[link_name = "llvm.s390.vgfmab"] fn vgfmab(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vgfmah"] fn vgfmah(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int;
    #[link_name = "llvm.s390.vgfmaf"] fn vgfmaf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long;
    #[link_name = "llvm.s390.vgfmag"] fn vgfmag(a: vector_unsigned_long_long, b: vector_unsigned_long_long, c: u128) -> u128;

    #[link_name = "llvm.s390.vbperm"] fn vbperm(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_long_long;

    #[link_name = "llvm.s390.vftcisb"] fn vftcisb(a: vector_float, b: u32) -> PackedTuple<vector_bool_int, i32>;
    #[link_name = "llvm.s390.vftcidb"] fn vftcidb(a: vector_double, b: u32) -> PackedTuple<vector_bool_long_long, i32>;

    #[link_name = "llvm.s390.vtm"] fn vtm(a: i8x16, b: i8x16) -> i32;

    #[link_name = "llvm.s390.vstrsb"] fn vstrsb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vstrsh"] fn vstrsh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vstrsf"] fn vstrsf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;

    #[link_name = "llvm.s390.vstrszb"] fn vstrszb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vstrszh"] fn vstrszh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vstrszf"] fn vstrszf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;

    #[link_name = "llvm.s390.vistrb"] fn vistrb(a: vector_unsigned_char) -> vector_unsigned_char;
    #[link_name = "llvm.s390.vistrh"] fn vistrh(a: vector_unsigned_short) -> vector_unsigned_short;
    #[link_name = "llvm.s390.vistrf"] fn vistrf(a: vector_unsigned_int) -> vector_unsigned_int;

    #[link_name = "llvm.s390.vistrbs"] fn vistrbs(a: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32>;
    #[link_name = "llvm.s390.vistrhs"] fn vistrhs(a: vector_unsigned_short) -> PackedTuple<vector_unsigned_short, i32>;
    #[link_name = "llvm.s390.vistrfs"] fn vistrfs(a: vector_unsigned_int) -> PackedTuple<vector_unsigned_int, i32>;

    #[link_name = "llvm.s390.vmslg"] fn vmslg(a: vector_unsigned_long_long, b: vector_unsigned_long_long, c: u128, d: u32) -> u128;

    #[link_name = "llvm.s390.vstrcb"] fn vstrcb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char, d: u32) -> vector_bool_char;
    #[link_name = "llvm.s390.vstrch"] fn vstrch(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short, d: u32) -> vector_bool_short;
    #[link_name = "llvm.s390.vstrcf"] fn vstrcf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int, d: u32) -> vector_bool_int;

    #[link_name = "llvm.s390.vstrcbs"] fn vstrcbs(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char, d: u32) -> PackedTuple<vector_bool_char, i32>;
    #[link_name = "llvm.s390.vstrchs"] fn vstrchs(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short, d: u32) -> PackedTuple<vector_bool_short, i32>;
    #[link_name = "llvm.s390.vstrcfs"] fn vstrcfs(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int, d: u32) -> PackedTuple<vector_bool_int, i32>;

    #[link_name = "llvm.s390.vstrczb"] fn vstrczb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char, d: u32) -> vector_bool_char;
    #[link_name = "llvm.s390.vstrczh"] fn vstrczh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short, d: u32) -> vector_bool_short;
    #[link_name = "llvm.s390.vstrczf"] fn vstrczf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int, d: u32) -> vector_bool_int;

    #[link_name = "llvm.s390.vstrczbs"] fn vstrczbs(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char, d: u32) -> PackedTuple<vector_bool_char, i32>;
    #[link_name = "llvm.s390.vstrczhs"] fn vstrczhs(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short, d: u32) -> PackedTuple<vector_bool_short, i32>;
    #[link_name = "llvm.s390.vstrczfs"] fn vstrczfs(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int, d: u32) -> PackedTuple<vector_bool_int, i32>;

    #[link_name = "llvm.s390.vfeeb"] fn vfeeb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.s390.vfeeh"] fn vfeeh(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.s390.vfeef"] fn vfeef(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.s390.vfeezb"] fn vfeezb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.s390.vfeezh"] fn vfeezh(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.s390.vfeezf"] fn vfeezf(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.s390.vfeebs"] fn vfeebs(a: i8x16, b: i8x16) -> PackedTuple<i8x16, i32>;
    #[link_name = "llvm.s390.vfeehs"] fn vfeehs(a: i16x8, b: i16x8) -> PackedTuple<i16x8, i32>;
    #[link_name = "llvm.s390.vfeefs"] fn vfeefs(a: i32x4, b: i32x4) -> PackedTuple<i32x4, i32>;

    #[link_name = "llvm.s390.vfeezbs"] fn vfeezbs(a: i8x16, b: i8x16) -> PackedTuple<i8x16, i32>;
    #[link_name = "llvm.s390.vfeezhs"] fn vfeezhs(a: i16x8, b: i16x8) -> PackedTuple<i16x8, i32>;
    #[link_name = "llvm.s390.vfeezfs"] fn vfeezfs(a: i32x4, b: i32x4) -> PackedTuple<i32x4, i32>;

    #[link_name = "llvm.s390.vfeneb"] fn vfeneb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.s390.vfeneh"] fn vfeneh(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.s390.vfenef"] fn vfenef(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.s390.vfenezb"] fn vfenezb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.s390.vfenezh"] fn vfenezh(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.s390.vfenezf"] fn vfenezf(a: i32x4, b: i32x4) -> i32x4;

    #[link_name = "llvm.s390.vfenebs"] fn vfenebs(a: i8x16, b: i8x16) -> PackedTuple<i8x16, i32>;
    #[link_name = "llvm.s390.vfenehs"] fn vfenehs(a: i16x8, b: i16x8) -> PackedTuple<i16x8, i32>;
    #[link_name = "llvm.s390.vfenefs"] fn vfenefs(a: i32x4, b: i32x4) -> PackedTuple<i32x4, i32>;

    #[link_name = "llvm.s390.vfenezbs"] fn vfenezbs(a: i8x16, b: i8x16) -> PackedTuple<i8x16, i32>;
    #[link_name = "llvm.s390.vfenezhs"] fn vfenezhs(a: i16x8, b: i16x8) -> PackedTuple<i16x8, i32>;
    #[link_name = "llvm.s390.vfenezfs"] fn vfenezfs(a: i32x4, b: i32x4) -> PackedTuple<i32x4, i32>;
}

impl_from! { i8x16, u8x16,  i16x8, u16x8, i32x4, u32x4, i64x2, u64x2, f32x4, f64x2 }

impl_neg! { i8x16 : 0 }
impl_neg! { i16x8 : 0 }
impl_neg! { i32x4 : 0 }
impl_neg! { i64x2 : 0 }
impl_neg! { f32x4 : 0f32 }
impl_neg! { f64x2 : 0f64 }

#[repr(simd)]
struct ShuffleMask<const N: usize>([u32; N]);

impl<const N: usize> ShuffleMask<N> {
    const fn reverse() -> Self {
        let mut index = [0; N];
        let mut i = 0;
        while i < N {
            index[i] = (N - i - 1) as u32;
            i += 1;
        }
        ShuffleMask(index)
    }

    const fn merge_low() -> Self {
        let mut mask = [0; N];
        let mut i = N / 2;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;
            mask[index + 1] = (i + N) as u32;

            i += 1;
            index += 2;
        }
        ShuffleMask(mask)
    }

    const fn merge_high() -> Self {
        let mut mask = [0; N];
        let mut i = 0;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;
            mask[index + 1] = (i + N) as u32;

            i += 1;
            index += 2;
        }
        ShuffleMask(mask)
    }

    const fn even() -> Self {
        let mut mask = [0; N];
        let mut i = 0;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;

            i += 2;
            index += 1;
        }
        ShuffleMask(mask)
    }

    const fn odd() -> Self {
        let mut mask = [0; N];
        let mut i = 1;
        let mut index = 0;
        while index < N {
            mask[index] = i as u32;

            i += 2;
            index += 1;
        }
        ShuffleMask(mask)
    }

    const fn pack() -> Self {
        Self::odd()
    }

    const fn unpack_low() -> Self {
        let mut mask = [0; N];
        let mut i = 0;
        while i < N {
            mask[i] = (N + i) as u32;
            i += 1;
        }
        ShuffleMask(mask)
    }

    const fn unpack_high() -> Self {
        let mut mask = [0; N];
        let mut i = 0;
        while i < N {
            mask[i] = i as u32;
            i += 1;
        }
        ShuffleMask(mask)
    }
}

const fn genmask<const MASK: u16>() -> [u8; 16] {
    let mut bits = MASK;
    let mut elements = [0u8; 16];

    let mut i = 0;
    while i < 16 {
        elements[i] = match bits & (1u16 << 15) {
            0 => 0,
            _ => 0xFF,
        };

        bits <<= 1;
        i += 1;
    }

    elements
}

const fn genmasks(bit_width: u32, a: u8, b: u8) -> u64 {
    let bit_width = bit_width as u8;
    let a = a % bit_width;
    let mut b = b % bit_width;
    if a > b {
        b = bit_width - 1;
    }

    // of course these indices start from the left
    let a = (bit_width - 1) - a;
    let b = (bit_width - 1) - b;

    ((1u64.wrapping_shl(a as u32 + 1)) - 1) & !((1u64.wrapping_shl(b as u32)) - 1)
}

const fn validate_block_boundary(block_boundary: u16) -> u32 {
    assert!(
        block_boundary.is_power_of_two() && block_boundary >= 64 && block_boundary <= 4096,
        "block boundary must be a constant power of 2 from 64 to 4096",
    );

    // so that 64 is encoded as 0, 128 as 1, ect.
    block_boundary as u32 >> 7
}

enum FindImm {
    Eq = 4,
    Ne = 12,
    EqIdx = 0,
    NeIdx = 8,
}

#[macro_use]
mod sealed {
    use super::*;

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAdd<Other> {
        type Result;
        unsafe fn vec_add(self, other: Other) -> Self::Result;
    }

    macro_rules! impl_add {
        ($name:ident, $a:ty, $instr:ident) => {
            impl_add!($name, $a, $a, $a, $instr);
        };
        ($name:ident, $a:ty, $b:ty, $c:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $b) -> $c {
                transmute(simd_add(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorAdd<$b> for $a {
                type Result = $c;

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_add(self, other: $b) -> Self::Result {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_add {
        use super::*;

        impl_add!(va_sc, vector_signed_char, vab);
        impl_add!(va_uc, vector_unsigned_char, vab);
        impl_add!(va_sh, vector_signed_short, vah);
        impl_add!(va_uh, vector_unsigned_short, vah);
        impl_add!(va_sf, vector_signed_int, vaf);
        impl_add!(va_uf, vector_unsigned_int, vaf);
        impl_add!(va_sg, vector_signed_long_long, vag);
        impl_add!(va_ug, vector_unsigned_long_long, vag);

        impl_add!(va_sc_bc, vector_signed_char, vector_bool_char, vector_signed_char, vab);
        impl_add!(va_uc_bc, vector_unsigned_char, vector_bool_char, vector_unsigned_char, vab);
        impl_add!(va_sh_bh, vector_signed_short, vector_bool_short, vector_signed_short, vah);
        impl_add!(va_uh_bh, vector_unsigned_short, vector_bool_short, vector_unsigned_short, vah);
        impl_add!(va_sf_bf, vector_signed_int, vector_bool_int, vector_signed_int, vaf);
        impl_add!(va_uf_bf, vector_unsigned_int, vector_bool_int, vector_unsigned_int, vaf);
        impl_add!(va_sg_bg, vector_signed_long_long, vector_bool_long_long, vector_signed_long_long, vag);
        impl_add!(va_ug_bg, vector_unsigned_long_long, vector_bool_long_long, vector_unsigned_long_long, vag);

        impl_add!(va_bc_sc, vector_bool_char, vector_signed_char, vector_signed_char, vab);
        impl_add!(va_bc_uc, vector_bool_char, vector_unsigned_char, vector_unsigned_char, vab);
        impl_add!(va_bh_sh, vector_bool_short, vector_signed_short, vector_signed_short, vah);
        impl_add!(va_bh_uh, vector_bool_short, vector_unsigned_short, vector_unsigned_short, vah);
        impl_add!(va_bf_sf, vector_bool_int, vector_signed_int, vector_signed_int, vaf);
        impl_add!(va_bf_uf, vector_bool_int, vector_unsigned_int, vector_unsigned_int, vaf);
        impl_add!(va_bg_sg, vector_bool_long_long, vector_signed_long_long, vector_signed_long_long, vag);
        impl_add!(va_bg_ug, vector_bool_long_long, vector_unsigned_long_long, vector_unsigned_long_long, vag);

        impl_add!(va_double, vector_double, vfadb);

        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vfasb))]
        pub unsafe fn va_float(a: vector_float, b: vector_float) -> vector_float {
            transmute(simd_add(a, b))
        }

        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl VectorAdd<Self> for vector_float {
            type Result = Self;

            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn vec_add(self, other: Self) -> Self::Result {
                va_float(self, other)
            }
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSub<Other> {
        type Result;
        unsafe fn vec_sub(self, other: Other) -> Self::Result;
    }

    macro_rules! impl_sub {
        ($name:ident, $a:ty, $instr:ident) => {
            impl_sub!($name, $a, $a, $a, $instr);
        };
        ($name:ident, $a:ty, $b:ty, $c:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $b) -> $c {
                transmute(simd_sub(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorSub<$b> for $a {
                type Result = $c;

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_sub(self, other: $b) -> Self::Result {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_sub {
        use super::*;

        impl_sub!(vs_sc, vector_signed_char, vsb);
        impl_sub!(vs_uc, vector_unsigned_char, vsb);
        impl_sub!(vs_sh, vector_signed_short, vsh);
        impl_sub!(vs_uh, vector_unsigned_short, vsh);
        impl_sub!(vs_sf, vector_signed_int, vsf);
        impl_sub!(vs_uf, vector_unsigned_int, vsf);
        impl_sub!(vs_sg, vector_signed_long_long, vsg);
        impl_sub!(vs_ug, vector_unsigned_long_long, vsg);

        impl_sub!(vs_sc_bc, vector_signed_char, vector_bool_char, vector_signed_char, vsb);
        impl_sub!(vs_uc_bc, vector_unsigned_char, vector_bool_char, vector_unsigned_char, vsb);
        impl_sub!(vs_sh_bh, vector_signed_short, vector_bool_short, vector_signed_short, vsh);
        impl_sub!(vs_uh_bh, vector_unsigned_short, vector_bool_short, vector_unsigned_short, vsh);
        impl_sub!(vs_sf_bf, vector_signed_int, vector_bool_int, vector_signed_int, vsf);
        impl_sub!(vs_uf_bf, vector_unsigned_int, vector_bool_int, vector_unsigned_int, vsf);
        impl_sub!(vs_sg_bg, vector_signed_long_long, vector_bool_long_long, vector_signed_long_long, vsg);
        impl_sub!(vs_ug_bg, vector_unsigned_long_long, vector_bool_long_long, vector_unsigned_long_long, vsg);

        impl_sub!(vs_bc_sc, vector_bool_char, vector_signed_char, vector_signed_char, vsb);
        impl_sub!(vs_bc_uc, vector_bool_char, vector_unsigned_char, vector_unsigned_char, vsb);
        impl_sub!(vs_bh_sh, vector_bool_short, vector_signed_short, vector_signed_short, vsh);
        impl_sub!(vs_bh_uh, vector_bool_short, vector_unsigned_short, vector_unsigned_short, vsh);
        impl_sub!(vs_bf_sf, vector_bool_int, vector_signed_int, vector_signed_int, vsf);
        impl_sub!(vs_bf_uf, vector_bool_int, vector_unsigned_int, vector_unsigned_int, vsf);
        impl_sub!(vs_bg_sg, vector_bool_long_long, vector_signed_long_long, vector_signed_long_long, vsg);
        impl_sub!(vs_bg_ug, vector_bool_long_long, vector_unsigned_long_long, vector_unsigned_long_long, vsg);

        impl_sub!(vs_double, vector_double, vfsdb);

        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vfssb))]
        pub unsafe fn vs_float(a: vector_float, b: vector_float) -> vector_float {
            transmute(simd_sub(a, b))
        }

        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl VectorSub<Self> for vector_float {
            type Result = Self;

            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn vec_sub(self, other: Self) -> Self::Result {
                vs_float(self, other)
            }
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMul {
        unsafe fn vec_mul(self, b: Self) -> Self;
    }

    macro_rules! impl_mul {
        ($name:ident, $a:ty, std_simd) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorMul for $a {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_mul(self, other: Self) -> Self {
                    transmute(simd_mul(transmute(self), other))
                }
            }
        };
        ($name:ident, $a:ty, $instr:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            pub unsafe fn $name(a: $a, b: $a) -> $a {
                transmute(simd_mul(transmute(a), b))
            }

            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorMul for $a {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_mul(self, other: Self) -> Self {
                    $name(self, other)
                }
            }
        };
    }

    #[rustfmt::skip]
    mod impl_mul {
        use super::*;

        impl_mul!(vml_sc, vector_signed_char, vmlb);
        impl_mul!(vml_uc, vector_unsigned_char, vmlb);
        impl_mul!(vml_sh, vector_signed_short, vmlhw);
        impl_mul!(vml_uh, vector_unsigned_short, vmlhw);
        impl_mul!(vml_sf, vector_signed_int, vmlf);
        impl_mul!(vml_uf, vector_unsigned_int, vmlf);
        impl_mul!(vml_sg, vector_signed_long_long, std_simd);
        impl_mul!(vml_ug, vector_unsigned_long_long, std_simd);

        impl_mul!(vml_float, vector_float, std_simd);
        impl_mul!(vml_double, vector_double, vfmdb);
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMax<Other> {
        type Result;
        unsafe fn vec_max(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vmxsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vmxb, vmxb] }
    test_impl! { vec_vmxsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vmxh, vmxh] }
    test_impl! { vec_vmxsf (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vmxf, vmxf] }
    test_impl! { vec_vmxsg (a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long [vmxg, vmxg] }

    test_impl! { vec_vmxslb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vmxlb, vmxlb] }
    test_impl! { vec_vmxslh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vmxlh, vmxlh] }
    test_impl! { vec_vmxslf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vmxlf, vmxlf] }
    test_impl! { vec_vmxslg (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vmxlg, vmxlg] }

    impl_vec_trait! { [VectorMax vec_max] ~(vmxlb, vmxb, vmxlh, vmxh, vmxlf, vmxf, vmxlg, vmxg) }

    test_impl! { vec_vfmaxsb (a: vector_float, b: vector_float) -> vector_float [simd_fmax, "vector-enhancements-1" vfmaxsb ] }
    test_impl! { vec_vfmaxdb (a: vector_double, b: vector_double) -> vector_double [simd_fmax, "vector-enhancements-1" vfmaxdb] }

    impl_vec_trait!([VectorMax vec_max] vec_vfmaxsb (vector_float, vector_float) -> vector_float);
    impl_vec_trait!([VectorMax vec_max] vec_vfmaxdb (vector_double, vector_double) -> vector_double);

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMin<Other> {
        type Result;
        unsafe fn vec_min(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vmnsb (a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [vmnb, vmnb] }
    test_impl! { vec_vmnsh (a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [vmnh, vmnh] }
    test_impl! { vec_vmnsf (a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [vmnf, vmnf] }
    test_impl! { vec_vmnsg (a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long [vmng, vmng] }

    test_impl! { vec_vmnslb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vmnlb, vmnlb] }
    test_impl! { vec_vmnslh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vmnlh, vmnlh] }
    test_impl! { vec_vmnslf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vmnlf, vmnlf] }
    test_impl! { vec_vmnslg (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vmnlg, vmnlg] }

    impl_vec_trait! { [VectorMin vec_min] ~(vmxlb, vmxb, vmxlh, vmxh, vmxlf, vmxf, vmxlg, vmxg) }

    test_impl! { vec_vfminsb (a: vector_float, b: vector_float) -> vector_float [simd_fmin, "vector-enhancements-1" vfminsb]  }
    test_impl! { vec_vfmindb (a: vector_double, b: vector_double) -> vector_double [simd_fmin, "vector-enhancements-1" vfmindb]  }

    impl_vec_trait!([VectorMin vec_min] vec_vfminsb (vector_float, vector_float) -> vector_float);
    impl_vec_trait!([VectorMin vec_min] vec_vfmindb (vector_double, vector_double) -> vector_double);

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAbs {
        unsafe fn vec_abs(self) -> Self;
    }

    macro_rules! impl_abs {
        ($name:ident, $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $name(v: s_t_l!($ty)) -> s_t_l!($ty) {
                v.vec_max(-v)
            }

            impl_vec_trait! { [VectorAbs vec_abs] $name (s_t_l!($ty)) }
        };
    }

    impl_abs! { vec_abs_i8, i8x16 }
    impl_abs! { vec_abs_i16, i16x8 }
    impl_abs! { vec_abs_i32, i32x4 }
    impl_abs! { vec_abs_i64, i64x2 }

    test_impl! { vec_abs_f32 (v: vector_float) -> vector_float [ simd_fabs, "vector-enhancements-1" vflpsb ] }
    test_impl! { vec_abs_f64 (v: vector_double) -> vector_double [ simd_fabs, vflpdb ] }

    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f32 (vector_float) }
    impl_vec_trait! { [VectorAbs vec_abs] vec_abs_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNabs {
        unsafe fn vec_nabs(self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-1"),
        assert_instr(vflnsb)
    )]
    unsafe fn vec_nabs_f32(a: vector_float) -> vector_float {
        simd_neg(simd_fabs(a))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vflndb))]
    unsafe fn vec_nabs_f64(a: vector_double) -> vector_double {
        simd_neg(simd_fabs(a))
    }

    impl_vec_trait! { [VectorNabs vec_nabs] vec_nabs_f32 (vector_float) }
    impl_vec_trait! { [VectorNabs vec_nabs] vec_nabs_f64 (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNmsub {
        unsafe fn vec_nmsub(self, b: Self, c: Self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vfnmssb)
    )]
    unsafe fn vec_nmsub_f32(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
        simd_neg(simd_fma(a, b, simd_neg(c)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorNmsub for vector_float {
        #[target_feature(enable = "vector")]
        unsafe fn vec_nmsub(self, b: Self, c: Self) -> Self {
            vec_nmsub_f32(self, b, c)
        }
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vfnmsdb)
    )]
    unsafe fn vec_nmsub_f64(a: vector_double, b: vector_double, c: vector_double) -> vector_double {
        simd_neg(simd_fma(a, b, simd_neg(c)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorNmsub for vector_double {
        #[target_feature(enable = "vector")]
        unsafe fn vec_nmsub(self, b: Self, c: Self) -> Self {
            vec_nmsub_f64(self, b, c)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNmadd {
        unsafe fn vec_nmadd(self, b: Self, c: Self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vfnmasb)
    )]
    unsafe fn vec_nmadd_f32(a: vector_float, b: vector_float, c: vector_float) -> vector_float {
        simd_neg(simd_fma(a, b, c))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorNmadd for vector_float {
        #[target_feature(enable = "vector")]
        unsafe fn vec_nmadd(self, b: Self, c: Self) -> Self {
            vec_nmadd_f32(self, b, c)
        }
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vfnmadb)
    )]
    unsafe fn vec_nmadd_f64(a: vector_double, b: vector_double, c: vector_double) -> vector_double {
        simd_neg(simd_fma(a, b, c))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorNmadd for vector_double {
        #[target_feature(enable = "vector")]
        unsafe fn vec_nmadd(self, b: Self, c: Self) -> Self {
            vec_nmadd_f64(self, b, c)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSplat {
        unsafe fn vec_splat<const IMM: u32>(self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepb, IMM2 = 1))]
    unsafe fn vrepb<const IMM2: u32>(a: vector_signed_char) -> vector_signed_char {
        static_assert_uimm_bits!(IMM2, 4);
        simd_shuffle(a, a, const { u32x16::from_array([IMM2; 16]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vreph, IMM2 = 1))]
    unsafe fn vreph<const IMM2: u32>(a: vector_signed_short) -> vector_signed_short {
        static_assert_uimm_bits!(IMM2, 3);
        simd_shuffle(a, a, const { u32x8::from_array([IMM2; 8]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepf, IMM2 = 1))]
    unsafe fn vrepf<const IMM2: u32>(a: vector_signed_int) -> vector_signed_int {
        static_assert_uimm_bits!(IMM2, 2);
        simd_shuffle(a, a, const { u32x4::from_array([IMM2; 4]) })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vrepg, IMM2 = 1))]
    unsafe fn vrepg<const IMM2: u32>(a: vector_signed_long_long) -> vector_signed_long_long {
        static_assert_uimm_bits!(IMM2, 1);
        simd_shuffle(a, a, const { u32x2::from_array([IMM2; 2]) })
    }

    macro_rules! impl_vec_splat {
        ($ty:ty, $fun:ident) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl VectorSplat for $ty {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_splat<const IMM: u32>(self) -> Self {
                    transmute($fun::<IMM>(transmute(self)))
                }
            }
        };
    }

    impl_vec_splat! { vector_signed_char, vrepb }
    impl_vec_splat! { vector_unsigned_char, vrepb }
    impl_vec_splat! { vector_bool_char, vrepb }
    impl_vec_splat! { vector_signed_short, vreph }
    impl_vec_splat! { vector_unsigned_short, vreph }
    impl_vec_splat! { vector_bool_short, vreph }
    impl_vec_splat! { vector_signed_int, vrepf }
    impl_vec_splat! { vector_unsigned_int, vrepf }
    impl_vec_splat! { vector_bool_int, vrepf }
    impl_vec_splat! { vector_signed_long_long, vrepg }
    impl_vec_splat! { vector_unsigned_long_long, vrepg }
    impl_vec_splat! { vector_bool_long_long, vrepg }

    impl_vec_splat! { vector_float, vrepf }
    impl_vec_splat! { vector_double, vrepg }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSplats<Output> {
        unsafe fn vec_splats(self) -> Output;
    }

    macro_rules! impl_vec_splats {
        ($(($fn:ident ($ty:ty, $shortty:tt) $instr:ident)),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($instr))]
                pub unsafe fn $fn(v: $ty) -> s_t_l!($shortty) {
                    transmute($shortty::splat(v))
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSplats<s_t_l!($shortty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_splats(self) -> s_t_l!($shortty) {
                        $fn (self)
                    }
                }
            )*
        }
    }

    impl_vec_splats! {
        (vec_splats_u8 (u8, u8x16) vrepb),
        (vec_splats_i8 (i8, i8x16) vrepb),
        (vec_splats_u16 (u16, u16x8) vreph),
        (vec_splats_i16 (i16, i16x8) vreph),
        (vec_splats_u32 (u32, u32x4) vrepf),
        (vec_splats_i32 (i32, i32x4) vrepf),
        (vec_splats_u64 (u64, u64x2) vlvgp),
        (vec_splats_i64 (i64, i64x2) vlvgp),
        (vec_splats_f32 (f32, f32x4) vrepf),
        (vec_splats_f64 (f64, f64x2) vrepg)
    }

    macro_rules! impl_bool_vec_splats {
        ($(($ty:ty, $shortty:tt, $boolty:ty)),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSplats<$boolty> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_splats(self) -> $boolty {
                        transmute($shortty::splat(self))
                    }
                }
            )*
        }
    }

    impl_bool_vec_splats! {
        (u8, u8x16, vector_bool_char),
        (i8, i8x16, vector_bool_char),
        (u16, u16x8, vector_bool_short),
        (i16, i16x8, vector_bool_short),
        (u32, u32x4, vector_bool_int),
        (i32, i32x4, vector_bool_int),
        (u64, u64x2, vector_bool_long_long),
        (i64, i64x2, vector_bool_long_long)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait CountBits {
        type Result;

        unsafe fn vec_cntlz(self) -> Self::Result;
        unsafe fn vec_cnttz(self) -> Self::Result;
        unsafe fn vec_popcnt(self) -> Self::Result;
    }

    macro_rules! impl_count_bits {
        ($ty:tt) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl CountBits for $ty {
                type Result = t_u!($ty);

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_cntlz(self) -> Self::Result {
                    transmute(simd_ctlz(self))
                }

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_cnttz(self) -> Self::Result {
                    transmute(simd_cttz(self))
                }

                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn vec_popcnt(self) -> Self::Result {
                    transmute(simd_ctpop(self))
                }
            }
        };
    }

    impl_count_bits!(vector_signed_char);
    impl_count_bits!(vector_unsigned_char);
    impl_count_bits!(vector_signed_short);
    impl_count_bits!(vector_unsigned_short);
    impl_count_bits!(vector_signed_int);
    impl_count_bits!(vector_unsigned_int);
    impl_count_bits!(vector_signed_long_long);
    impl_count_bits!(vector_unsigned_long_long);

    test_impl! { vec_clzb_signed +(a: vector_signed_char) -> vector_unsigned_char [simd_ctlz, vclzb] }
    test_impl! { vec_clzh_signed +(a: vector_signed_short) -> vector_unsigned_short [simd_ctlz, vclzh] }
    test_impl! { vec_clzf_signed +(a: vector_signed_int) -> vector_unsigned_int [simd_ctlz, vclzf] }
    test_impl! { vec_clzg_signed +(a: vector_signed_long_long) -> vector_unsigned_long_long [simd_ctlz, vclzg] }

    test_impl! { vec_clzb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_ctlz, vclzb] }
    test_impl! { vec_clzh_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_ctlz, vclzh] }
    test_impl! { vec_clzf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_ctlz, vclzf] }
    test_impl! { vec_clzg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_ctlz, vclzg] }

    test_impl! { vec_ctzb_signed +(a: vector_signed_char) -> vector_unsigned_char [simd_cttz, vctzb] }
    test_impl! { vec_ctzh_signed +(a: vector_signed_short) -> vector_unsigned_short [simd_cttz, vctzh] }
    test_impl! { vec_ctzf_signed +(a: vector_signed_int) -> vector_unsigned_int [simd_cttz, vctzf] }
    test_impl! { vec_ctzg_signed +(a: vector_signed_long_long) -> vector_unsigned_long_long [simd_cttz, vctzg] }

    test_impl! { vec_ctzb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_cttz, vctzb] }
    test_impl! { vec_ctzh_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_cttz, vctzh] }
    test_impl! { vec_ctzf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_cttz, vctzf] }
    test_impl! { vec_ctzg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_cttz, vctzg] }

    test_impl! { vec_vpopctb_signed +(a: vector_signed_char) -> vector_signed_char [simd_ctpop, vpopctb] }
    test_impl! { vec_vpopcth_signed +(a: vector_signed_short) -> vector_signed_short [simd_ctpop, "vector-enhancements-1" vpopcth] }
    test_impl! { vec_vpopctf_signed +(a: vector_signed_int) -> vector_signed_int [simd_ctpop, "vector-enhancements-1" vpopctf] }
    test_impl! { vec_vpopctg_signed +(a: vector_signed_long_long) -> vector_signed_long_long [simd_ctpop, "vector-enhancements-1" vpopctg] }

    test_impl! { vec_vpopctb_unsigned +(a: vector_unsigned_char) -> vector_unsigned_char [simd_ctpop, vpopctb] }
    test_impl! { vec_vpopcth_unsigned +(a: vector_unsigned_short) -> vector_unsigned_short [simd_ctpop, "vector-enhancements-1" vpopcth] }
    test_impl! { vec_vpopctf_unsigned +(a: vector_unsigned_int) -> vector_unsigned_int [simd_ctpop, "vector-enhancements-1" vpopctf] }
    test_impl! { vec_vpopctg_unsigned +(a: vector_unsigned_long_long) -> vector_unsigned_long_long [simd_ctpop, "vector-enhancements-1" vpopctg] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAnd<Other> {
        type Result;
        unsafe fn vec_and(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAnd vec_and] ~(simd_and) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorOr<Other> {
        type Result;
        unsafe fn vec_or(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorOr vec_or] ~(simd_or) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorXor<Other> {
        type Result;
        unsafe fn vec_xor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorXor vec_xor] ~(simd_xor) }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vno))]
    unsafe fn nor(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_or(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNor<Other> {
        type Result;
        unsafe fn vec_nor(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorNor vec_nor]+ 2c (nor) }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnn))]
    unsafe fn nand(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_and(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorNand<Other> {
        type Result;
        unsafe fn vec_nand(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorNand vec_nand]+ 2c (nand) }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnx))]
    unsafe fn eqv(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a: u8x16 = transmute(a);
        let b: u8x16 = transmute(b);
        transmute(simd_xor(simd_xor(a, b), u8x16::splat(0xff)))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorEqv<Other> {
        type Result;
        unsafe fn vec_eqv(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorEqv vec_eqv]+ 2c (eqv) }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vnc))]
    unsafe fn andc(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a = transmute(a);
        let b = transmute(b);
        transmute(simd_and(simd_xor(u8x16::splat(0xff), b), a))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAndc<Other> {
        type Result;
        unsafe fn vec_andc(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAndc vec_andc]+ 2c (andc) }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(voc))]
    unsafe fn orc(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char {
        let a = transmute(a);
        let b = transmute(b);
        transmute(simd_or(simd_xor(u8x16::splat(0xff), b), a))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorOrc<Other> {
        type Result;
        unsafe fn vec_orc(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorOrc vec_orc]+ 2c (orc) }

    // Z vector intrinsic      C23 math.h  LLVM IR         ISO/IEC 60559 operation        inexact  vfidb parameters
    //
    // vec_rint                rint        llvm.rint       roundToIntegralExact           yes      0, 0
    // vec_roundc              nearbyint   llvm.nearbyint  n/a                            no       4, 0
    // vec_floor / vec_roundm  floor       llvm.floor      roundToIntegralTowardNegative  no       4, 7
    // vec_ceil / vec_roundp   ceil        llvm.ceil       roundToIntegralTowardPositive  no       4, 6
    // vec_trunc / vec_roundz  trunc       llvm.trunc      roundToIntegralTowardZero      no       4, 5
    // vec_round               roundeven   llvm.roundeven  roundToIntegralTiesToEven      no       4, 4
    // n/a                     round       llvm.round      roundToIntegralTiesAway        no       4, 1

    // `simd_round_ties_even` is implemented as `llvm.rint`.
    test_impl! { vec_rint_f32 (a: vector_float) -> vector_float [simd_round_ties_even, "vector-enhancements-1" vfisb] }
    test_impl! { vec_rint_f64 (a: vector_double) -> vector_double [simd_round_ties_even, vfidb] }

    test_impl! { vec_roundc_f32 (a: vector_float) -> vector_float [nearbyint_v4f32,  "vector-enhancements-1" vfisb] }
    test_impl! { vec_roundc_f64 (a: vector_double) -> vector_double [nearbyint_v2f64, vfidb] }

    test_impl! { vec_round_f32 (a: vector_float) -> vector_float [roundeven_v4f32, "vector-enhancements-1" vfisb] }
    test_impl! { vec_round_f64 (a: vector_double) -> vector_double [roundeven_v2f64, vfidb] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRoundc {
        unsafe fn vec_roundc(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRound {
        unsafe fn vec_round(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRint {
        unsafe fn vec_rint(self) -> Self;
    }

    impl_vec_trait! { [VectorRoundc vec_roundc] vec_roundc_f32 (vector_float) }
    impl_vec_trait! { [VectorRoundc vec_roundc] vec_roundc_f64 (vector_double) }

    impl_vec_trait! { [VectorRound vec_round] vec_round_f32 (vector_float) }
    impl_vec_trait! { [VectorRound vec_round] vec_round_f64 (vector_double) }

    impl_vec_trait! { [VectorRint vec_rint] simd_round_ties_even (vector_float) }
    impl_vec_trait! { [VectorRint vec_rint] simd_round_ties_even (vector_double) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorTrunc {
        // same as vec_roundz
        unsafe fn vec_trunc(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCeil {
        // same as vec_roundp
        unsafe fn vec_ceil(self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFloor {
        // same as vec_roundm
        unsafe fn vec_floor(self) -> Self;
    }

    impl_vec_trait! { [VectorTrunc vec_trunc] simd_trunc (vector_float) }
    impl_vec_trait! { [VectorTrunc vec_trunc] simd_trunc (vector_double) }

    impl_vec_trait! { [VectorCeil vec_ceil] simd_ceil (vector_float) }
    impl_vec_trait! { [VectorCeil vec_ceil] simd_ceil (vector_double) }

    impl_vec_trait! { [VectorFloor vec_floor] simd_floor (vector_float) }
    impl_vec_trait! { [VectorFloor vec_floor] simd_floor (vector_double) }

    macro_rules! impl_vec_shift {
        ([$Trait:ident $m:ident] ($b:ident, $h:ident, $w:ident, $g:ident)) => {
            impl_vec_trait!{ [$Trait $m]+ $b (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$Trait $m]+ $b (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$Trait $m]+ $h (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$Trait $m]+ $h (vector_signed_short, vector_unsigned_short) -> vector_signed_short }
            impl_vec_trait!{ [$Trait $m]+ $w (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$Trait $m]+ $w (vector_signed_int, vector_unsigned_int) -> vector_signed_int }
            impl_vec_trait!{ [$Trait $m]+ $g (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$Trait $m]+ $g (vector_signed_long_long, vector_unsigned_long_long) -> vector_signed_long_long }
        };
    }

    macro_rules! impl_shift {
        ($fun:ident $intr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($fun))]
            unsafe fn $fun(a: t_t_l!($ty), b: t_t_l!($ty)) -> t_t_l!($ty) {
                let a = transmute(a);
                // use the remainder of b by the width of a's elements to prevent UB
                let b = simd_rem(transmute(b), <t_t_s!($ty)>::splat($ty::BITS as $ty));

                transmute($intr(a, b))
            }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSl<Other> {
        type Result;
        unsafe fn vec_sl(self, b: Other) -> Self::Result;
    }

    impl_shift! { veslvb simd_shl u8 }
    impl_shift! { veslvh simd_shl u16 }
    impl_shift! { veslvf simd_shl u32 }
    impl_shift! { veslvg simd_shl u64 }

    impl_vec_shift! { [VectorSl vec_sl] (veslvb, veslvh, veslvf, veslvg) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSr<Other> {
        type Result;
        unsafe fn vec_sr(self, b: Other) -> Self::Result;
    }

    impl_shift! { vesrlvb simd_shr u8 }
    impl_shift! { vesrlvh simd_shr u16 }
    impl_shift! { vesrlvf simd_shr u32 }
    impl_shift! { vesrlvg simd_shr u64 }

    impl_vec_shift! { [VectorSr vec_sr] (vesrlvb, vesrlvh, vesrlvf, vesrlvg) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSra<Other> {
        type Result;
        unsafe fn vec_sra(self, b: Other) -> Self::Result;
    }

    impl_shift! { vesravb simd_shr i8 }
    impl_shift! { vesravh simd_shr i16 }
    impl_shift! { vesravf simd_shr i32 }
    impl_shift! { vesravg simd_shr i64 }

    impl_vec_shift! { [VectorSra vec_sra] (vesravb, vesravh, vesravf, vesravg) }

    macro_rules! impl_vec_shift_byte {
        ([$trait:ident $m:ident] ($f:ident)) => {
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_signed_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_signed_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_signed_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_signed_short) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_unsigned_short) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_signed_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_signed_int) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_unsigned_int) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_signed_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_signed_long_long) -> vector_signed_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_unsigned_long_long) -> vector_signed_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_float, vector_signed_int) -> vector_float }
            impl_vec_trait!{ [$trait $m]+ $f (vector_float, vector_unsigned_int) -> vector_float }
            impl_vec_trait!{ [$trait $m]+ $f (vector_double, vector_signed_long_long) -> vector_double }
            impl_vec_trait!{ [$trait $m]+ $f (vector_double, vector_unsigned_long_long) -> vector_double }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSlb<Other> {
        type Result;
        unsafe fn vec_slb(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSlb vec_slb] (vslb) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrab<Other> {
        type Result;
        unsafe fn vec_srab(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSrab vec_srab] (vsrab) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrb<Other> {
        type Result;
        unsafe fn vec_srb(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_byte! { [VectorSrb vec_srb] (vsrlb) }

    macro_rules! impl_vec_shift_long {
        ([$trait:ident $m:ident] ($f:ident)) => {
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_char, vector_unsigned_char) -> vector_signed_char }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_short, vector_unsigned_char) -> vector_unsigned_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_short, vector_unsigned_char) -> vector_signed_short }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_int, vector_unsigned_char) -> vector_unsigned_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_int, vector_unsigned_char) -> vector_signed_int }
            impl_vec_trait!{ [$trait $m]+ $f (vector_unsigned_long_long, vector_unsigned_char) -> vector_unsigned_long_long }
            impl_vec_trait!{ [$trait $m]+ $f (vector_signed_long_long, vector_unsigned_char) -> vector_signed_long_long }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrl<Other> {
        type Result;
        unsafe fn vec_srl(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSrl vec_srl] (vsrl) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSral<Other> {
        type Result;
        unsafe fn vec_sral(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSral vec_sral] (vsra) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSll<Other> {
        type Result;
        unsafe fn vec_sll(self, b: Other) -> Self::Result;
    }

    impl_vec_shift_long! { [VectorSll vec_sll] (vsl) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRl<Other> {
        type Result;
        unsafe fn vec_rl(self, b: Other) -> Self::Result;
    }

    macro_rules! impl_rot {
        ($fun:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($fun))]
            unsafe fn $fun(a: t_t_l!($ty), b: t_t_l!($ty)) -> t_t_l!($ty) {
                simd_funnel_shl(a, a, b)
            }
        };
    }

    impl_rot! { verllvb u8 }
    impl_rot! { verllvh u16 }
    impl_rot! { verllvf u32 }
    impl_rot! { verllvg u64 }

    impl_vec_shift! { [VectorRl vec_rl] (verllvb, verllvh, verllvf, verllvg) }

    macro_rules! test_rot_imm {
        ($fun:ident $instr:ident $ty:ident) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn $fun(a: t_t_l!($ty), bits: core::ffi::c_ulong) -> t_t_l!($ty) {
                // mod by the number of bits in a's element type to prevent UB
                let bits = (bits % $ty::BITS as core::ffi::c_ulong) as $ty;
                let b = <t_t_s!($ty)>::splat(bits);

                simd_funnel_shl(a, a, transmute(b))
            }
        };
    }

    test_rot_imm! { verllvb_imm verllb u8 }
    test_rot_imm! { verllvh_imm verllh u16 }
    test_rot_imm! { verllvf_imm verllf u32 }
    test_rot_imm! { verllvg_imm verllg u64 }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRli {
        unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self;
    }

    macro_rules! impl_rot_imm {
        ($($ty:ident, $intr:ident),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRli for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self {
                        transmute($intr(transmute(self), bits))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRli for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rli(self, bits: core::ffi::c_ulong) -> Self {
                        $intr(self, bits)
                    }
                }
            )*
        }
    }

    impl_rot_imm! {
        vector_signed_char, verllvb_imm,
        vector_signed_short, verllvh_imm,
        vector_signed_int, verllvf_imm,
        vector_signed_long_long, verllvg_imm
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRlMask<Other> {
        unsafe fn vec_rl_mask<const IMM8: u8>(self, other: Other) -> Self;
    }

    macro_rules! impl_rl_mask {
        ($($ty:ident, $intr:ident, $fun:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($intr, IMM8 = 6))]
                unsafe fn $fun<const IMM8: u8>(a: $ty, b: t_u!($ty)) -> $ty {
                    // mod by the number of bits in a's element type to prevent UB
                    $intr(a, a, transmute(b), const { (IMM8 % <l_t_t!($ty)>::BITS as u8) as i32 })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRlMask<t_u!($ty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rl_mask<const IMM8: u8>(self, other: t_u!($ty)) -> Self {
                        $fun::<IMM8>(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorRlMask<t_u!($ty)> for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_rl_mask<const IMM8: u8>(self, other: t_u!($ty)) -> Self {
                        transmute($fun::<IMM8>(transmute(self), transmute(other)))
                    }
                }
            )*
        }
    }

    impl_rl_mask! {
        vector_signed_char, verimb, test_verimb,
        vector_signed_short, verimh, test_verimh,
        vector_signed_int, verimf, test_verimf,
        vector_signed_long_long, verimg, test_verimg
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorReve {
        unsafe fn vec_reve(self) -> Self;
    }

    #[repr(simd)]
    struct ReverseMask<const N: usize>([u32; N]);

    impl<const N: usize> ReverseMask<N> {
        const fn new() -> Self {
            let mut index = [0; N];
            let mut i = 0;
            while i < N {
                index[i] = (N - i - 1) as u32;
                i += 1;
            }
            ReverseMask(index)
        }
    }

    macro_rules! impl_reve {
        ($($ty:ident, $fun:ident, $instr:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($instr))]
                unsafe fn $fun(a: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, a, const { ShuffleMask::<N>::reverse() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        $fun(self)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        transmute($fun(transmute(self)))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorReve for t_b!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_reve(self) -> Self {
                        transmute($fun(transmute(self)))
                    }
                }
            )*
        }
    }

    impl_reve! {
        vector_signed_char, reveb, vperm,
        vector_signed_short, reveh, vperm,
        vector_signed_int, revef, vperm,
        vector_signed_long_long, reveg, vpdi
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorReve for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_reve(self) -> Self {
            transmute(transmute::<_, vector_signed_int>(self).vec_reve())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorReve for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_reve(self) -> Self {
            transmute(transmute::<_, vector_signed_long_long>(self).vec_reve())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorRevb {
        unsafe fn vec_revb(self) -> Self;
    }

    test_impl! { bswapb (a: vector_signed_char) -> vector_signed_char [simd_bswap, _] }
    test_impl! { bswaph (a: vector_signed_short) -> vector_signed_short [simd_bswap, vperm] }
    test_impl! { bswapf (a: vector_signed_int) -> vector_signed_int [simd_bswap, vperm] }
    test_impl! { bswapg (a: vector_signed_long_long) -> vector_signed_long_long [simd_bswap, vperm] }

    impl_vec_trait! { [VectorRevb vec_revb]+ bswapb (vector_unsigned_char) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapb (vector_signed_char) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswaph (vector_unsigned_short) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswaph (vector_signed_short) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapf (vector_unsigned_int) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapf (vector_signed_int) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapg (vector_unsigned_long_long) }
    impl_vec_trait! { [VectorRevb vec_revb]+ bswapg (vector_signed_long_long) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRevb for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_revb(self) -> Self {
            transmute(transmute::<_, vector_signed_int>(self).vec_revb())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorRevb for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_revb(self) -> Self {
            transmute(transmute::<_, vector_signed_long_long>(self).vec_revb())
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMergel {
        unsafe fn vec_mergel(self, other: Self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMergeh {
        unsafe fn vec_mergeh(self, other: Self) -> Self;
    }

    macro_rules! impl_merge {
        ($($ty:ident, $mergel:ident, $mergeh:ident),*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($mergel))]
                unsafe fn $mergel(a: $ty, b: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, b, const { ShuffleMask::<N>::merge_low() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergel for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergel(self, other: Self) -> Self {
                        $mergel(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergel for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergel(self, other: Self) -> Self {
                        transmute($mergel(transmute(self), transmute(other)))
                    }
                }

                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($mergeh))]
                unsafe fn $mergeh(a: $ty, b: $ty) -> $ty {
                    const N: usize = core::mem::size_of::<$ty>() / core::mem::size_of::<l_t_t!($ty)>();
                    simd_shuffle(a, b, const { ShuffleMask::<N>::merge_high() })
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergeh for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergeh(self, other: Self) -> Self {
                        $mergeh(self, other)
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorMergeh for t_u!($ty) {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_mergeh(self, other: Self) -> Self {
                        transmute($mergeh(transmute(self), transmute(other)))
                    }
                }
            )*
        }
    }

    impl_merge! {
        vector_signed_char, vmrlb, vmrhb,
        vector_signed_short, vmrlh, vmrhh,
        vector_signed_int, vmrlf, vmrhf,
        vector_signed_long_long, vmrlg, vmrhg
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorPerm {
        unsafe fn vec_perm(self, other: Self, c: vector_unsigned_char) -> Self;
    }

    macro_rules! impl_merge {
        ($($ty:ident),*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorPerm for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_perm(self, other: Self, c: vector_unsigned_char) -> Self {
                        transmute(vperm(transmute(self), transmute(other), c))
                    }
                }
            )*
        }
    }

    impl_merge! {
        vector_signed_char,
        vector_signed_short,
        vector_signed_int,
        vector_signed_long_long,
        vector_unsigned_char,
        vector_unsigned_short,
        vector_unsigned_int,
        vector_unsigned_long_long,
        vector_bool_char,
        vector_bool_short,
        vector_bool_int,
        vector_bool_long_long,
        vector_float,
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSumU128 {
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsumqf))]
    pub unsafe fn vec_vsumqf(a: vector_unsigned_int, b: vector_unsigned_int) -> u128 {
        transmute(vsumqf(a, b))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsumqg))]
    pub unsafe fn vec_vsumqg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> u128 {
        transmute(vsumqg(a, b))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSumU128 for vector_unsigned_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char {
            transmute(vec_vsumqf(self, other))
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSumU128 for vector_unsigned_long_long {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum_u128(self, other: Self) -> vector_unsigned_char {
            transmute(vec_vsumqg(self, other))
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSum2 {
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long;
    }

    test_impl! { vec_vsumgh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_long_long [vsumgh, vsumgh] }
    test_impl! { vec_vsumgf (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long [vsumgf, vsumgf] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum2 for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long {
            vec_vsumgh(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum2 for vector_unsigned_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum2(self, other: Self) -> vector_unsigned_long_long {
            vec_vsumgf(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSum4 {
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int;
    }

    test_impl! { vec_vsumb (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_int [vsumb, vsumb] }
    test_impl! { vec_vsumh (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int [vsumh, vsumh] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum4 for vector_unsigned_char {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int {
            vec_vsumb(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorSum4 for vector_unsigned_short {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_sum4(self, other: Self) -> vector_unsigned_int {
            vec_vsumh(self, other)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSubc<Other> {
        type Result;
        unsafe fn vec_subc(self, b: Other) -> Self::Result;
    }

    test_impl! { vec_vscbib (a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [vscbib, vscbib] }
    test_impl! { vec_vscbih (a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [vscbih, vscbih] }
    test_impl! { vec_vscbif (a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [vscbif, vscbif] }
    test_impl! { vec_vscbig (a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [vscbig, vscbig] }

    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbib (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbih (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbif (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
    impl_vec_trait! {[VectorSubc vec_subc] vec_vscbig (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSqrt {
        unsafe fn vec_sqrt(self) -> Self;
    }

    test_impl! { vec_sqrt_f32 (v: vector_float) -> vector_float [ simd_fsqrt, "vector-enhancements-1" vfsqsb ] }
    test_impl! { vec_sqrt_f64 (v: vector_double) -> vector_double [ simd_fsqrt, vfsqdb ] }

    impl_vec_trait! { [VectorSqrt vec_sqrt] vec_sqrt_f32 (vector_float) }
    impl_vec_trait! { [VectorSqrt vec_sqrt] vec_sqrt_f64 (vector_double) }

    macro_rules! vfae_wrapper {
        ($($name:ident $ty:ident)*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($name, IMM = 0))]
                unsafe fn $name<const IMM: i32>(
                    a: $ty,
                    b: $ty,
                ) -> $ty {
                    super::$name(a, b, IMM)
                }
            )*
        }
     }

    vfae_wrapper! {
       vfaeb vector_signed_char
       vfaeh vector_signed_short
       vfaef vector_signed_int

       vfaezb vector_signed_char
       vfaezh vector_signed_short
       vfaezf vector_signed_int
    }

    macro_rules! impl_vfae {
        ([idx_cc $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [idx_cc $Trait $m] $imm
                $b vector_signed_char vector_signed_char
                $b vector_unsigned_char vector_unsigned_char
                $b vector_bool_char vector_unsigned_char

                $h vector_signed_short vector_signed_short
                $h vector_unsigned_short vector_unsigned_short
                $h vector_bool_short vector_unsigned_short

                $f vector_signed_int vector_signed_int
                $f vector_unsigned_int vector_unsigned_int
                $f vector_bool_int vector_unsigned_int
            }
        };
        ([idx_cc $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident $r:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = $r;
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b));
                        (transmute(x), y)
                    }
                }
            )*
        };
        ([cc $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [cc $Trait $m] $imm
                $b vector_signed_char
                $b vector_unsigned_char
                $b vector_bool_char

                $h vector_signed_short
                $h vector_unsigned_short
                $h vector_bool_short

                $f vector_signed_int
                $f vector_unsigned_int
                $f vector_bool_int
            }
        };
        ([cc $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = t_b!($ty);
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b));
                        (transmute(x), y)
                    }
                }
            )*
        };
        ([idx $Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [idx $Trait $m] $imm
                $b vector_signed_char vector_signed_char
                $b vector_unsigned_char vector_unsigned_char
                $b vector_bool_char vector_unsigned_char

                $h vector_signed_short vector_signed_short
                $h vector_unsigned_short vector_unsigned_short
                $h vector_bool_short vector_unsigned_short

                $f vector_signed_int vector_signed_int
                $f vector_unsigned_int vector_unsigned_int
                $f vector_bool_int vector_unsigned_int
            }
        };
        ([idx $Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident $r:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = $r;
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> Self::Result {
                        transmute($fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b)))
                    }
                }
            )*
        };
        ([$Trait:ident $m:ident] $imm:ident $b:ident $h:ident $f:ident) => {
            impl_vfae! { [$Trait $m] $imm
                $b vector_signed_char
                $b vector_unsigned_char
                $b vector_bool_char

                $h vector_signed_short
                $h vector_unsigned_short
                $h vector_bool_short

                $f vector_signed_int
                $f vector_unsigned_int
                $f vector_bool_int
            }
        };
        ([$Trait:ident $m:ident] $imm:ident $($fun:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl $Trait<Self> for $ty {
                    type Result = t_b!($ty);
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn $m(self, b: Self) -> Self::Result {
                        transmute($fun::<{ FindImm::$imm as i32 }>(transmute(self), transmute(b)))
                    }
                }
            )*
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEq<Other> {
        type Result;
        unsafe fn vec_find_any_eq(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [VectorFindAnyEq vec_find_any_eq] Eq vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNe<Other> {
        type Result;
        unsafe fn vec_find_any_ne(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [VectorFindAnyNe vec_find_any_ne] Ne vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqOrZeroIdx<Other> {
        type Result;
        unsafe fn vec_find_any_eq_or_0_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyEqOrZeroIdx vec_find_any_eq_or_0_idx] EqIdx
        vfaezb vector_signed_char vector_signed_char
        vfaezb vector_unsigned_char vector_unsigned_char
        vfaezb vector_bool_char vector_unsigned_char

        vfaezh vector_signed_short vector_signed_short
        vfaezh vector_unsigned_short vector_unsigned_short
        vfaezh vector_bool_short vector_unsigned_short

        vfaezf vector_signed_int vector_signed_int
        vfaezf vector_unsigned_int vector_unsigned_int
        vfaezf vector_bool_int vector_unsigned_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeOrZeroIdx<Other> {
        type Result;
        unsafe fn vec_find_any_ne_or_0_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyNeOrZeroIdx vec_find_any_ne_or_0_idx] NeIdx
        vfaezb vector_signed_char vector_signed_char
        vfaezb vector_unsigned_char vector_unsigned_char
        vfaezb vector_bool_char vector_unsigned_char

        vfaezh vector_signed_short vector_signed_short
        vfaezh vector_unsigned_short vector_unsigned_short
        vfaezh vector_bool_short vector_unsigned_short

        vfaezf vector_signed_int vector_signed_int
        vfaezf vector_unsigned_int vector_unsigned_int
        vfaezf vector_bool_int vector_unsigned_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqIdx<Other> {
        type Result;
        unsafe fn vec_find_any_eq_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyEqIdx vec_find_any_eq_idx] EqIdx vfaeb vfaeh vfaef }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeIdx<Other> {
        type Result;
        unsafe fn vec_find_any_ne_idx(self, other: Other) -> Self::Result;
    }

    impl_vfae! { [idx VectorFindAnyNeIdx vec_find_any_ne_idx] NeIdx vfaeb vfaeh vfaef }

    macro_rules! vfaes_wrapper {
        ($($name:ident $ty:ident)*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($name, IMM = 0))]
                unsafe fn $name<const IMM: i32>(
                    a: $ty,
                    b: $ty,
                ) -> PackedTuple<$ty, i32> {
                    super::$name(a, b, IMM)
                }
            )*
        }
     }

    vfaes_wrapper! {
        vfaebs vector_signed_char
        vfaehs vector_signed_short
        vfaefs vector_signed_int

        vfaezbs vector_signed_char
        vfaezhs vector_signed_short
        vfaezfs vector_signed_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [cc VectorFindAnyEqCC vec_find_any_eq_cc] Eq vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [cc VectorFindAnyNeCC vec_find_any_ne_cc] Ne vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_idx_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [idx_cc VectorFindAnyEqIdxCC vec_find_any_eq_idx_cc] EqIdx vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_idx_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [idx_cc VectorFindAnyNeIdxCC vec_find_any_ne_idx_cc] NeIdx vfaebs vfaehs vfaefs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyEqOrZeroIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_eq_or_0_idx_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [idx_cc VectorFindAnyEqOrZeroIdxCC vec_find_any_eq_or_0_idx_cc] EqIdx vfaezbs vfaezhs vfaezfs }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFindAnyNeOrZeroIdxCC<Other> {
        type Result;
        unsafe fn vec_find_any_ne_or_0_idx_cc(self, other: Other) -> (Self::Result, i32);
    }

    impl_vfae! { [idx_cc VectorFindAnyNeOrZeroIdxCC vec_find_any_ne_or_0_idx_cc] NeIdx vfaezbs vfaezhs vfaezfs }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vl))]
    unsafe fn test_vector_load(offset: isize, ptr: *const i32) -> vector_signed_int {
        ptr.byte_offset(offset)
            .cast::<vector_signed_int>()
            .read_unaligned()
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vst))]
    unsafe fn test_vector_store(vector: vector_signed_int, offset: isize, ptr: *mut i32) {
        ptr.byte_offset(offset)
            .cast::<vector_signed_int>()
            .write_unaligned(vector)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorLoad: Sized {
        type ElementType;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_xl(offset: isize, ptr: *const Self::ElementType) -> Self {
            ptr.byte_offset(offset).cast::<Self>().read_unaligned()
        }

        unsafe fn vec_load_len(ptr: *const Self::ElementType, byte_count: u32) -> Self;

        unsafe fn vec_load_bndry<const BLOCK_BOUNDARY: u16>(
            ptr: *const Self::ElementType,
        ) -> MaybeUninit<Self>;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorStore: Sized {
        type ElementType;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_xst(self, offset: isize, ptr: *mut Self::ElementType) {
            ptr.byte_offset(offset).cast::<Self>().write_unaligned(self)
        }

        unsafe fn vec_store_len(self, ptr: *mut Self::ElementType, byte_count: u32);
    }

    macro_rules! impl_load_store {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorLoad for t_t_l!($ty) {
                    type ElementType = $ty;

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_load_len(ptr: *const Self::ElementType, byte_count: u32) -> Self {
                        transmute(vll( byte_count, ptr.cast(),))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_load_bndry<const BLOCK_BOUNDARY: u16>(ptr: *const Self::ElementType) -> MaybeUninit<Self> {
                        transmute(vlbb(ptr.cast(), const { validate_block_boundary(BLOCK_BOUNDARY) }))
                    }

                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorStore for t_t_l!($ty) {
                    type ElementType = $ty;

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_store_len(self, ptr: *mut Self::ElementType, byte_count: u32) {
                        vstl(transmute(self), byte_count, ptr.cast())
                    }
                }
            )*
        }
    }

    impl_load_store! { i8 u8 i16 u16 i32 u32 i64 u64 f32 f64 }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vll))]
    unsafe fn test_vec_load_len(ptr: *const i32, byte_count: u32) -> vector_signed_int {
        vector_signed_int::vec_load_len(ptr, byte_count)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlbb))]
    unsafe fn test_vec_load_bndry(ptr: *const i32) -> MaybeUninit<vector_signed_int> {
        vector_signed_int::vec_load_bndry::<512>(ptr)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vstl))]
    unsafe fn test_vec_store_len(vector: vector_signed_int, ptr: *mut i32, byte_count: u32) {
        vector.vec_store_len(ptr, byte_count)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorLoadPair: Sized {
        type ElementType;

        unsafe fn vec_load_pair(a: Self::ElementType, b: Self::ElementType) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorLoadPair for vector_signed_long_long {
        type ElementType = i64;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_load_pair(a: i64, b: i64) -> Self {
            vector_signed_long_long([a, b])
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorLoadPair for vector_unsigned_long_long {
        type ElementType = u64;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_load_pair(a: u64, b: u64) -> Self {
            vector_unsigned_long_long([a, b])
        }
    }

    #[inline]
    #[target_feature(enable = "vector")]
    unsafe fn pack<T, const N: usize>(a: T, b: T) -> T {
        simd_shuffle(a, b, const { ShuffleMask::<N>::pack() })
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpkh))]
    unsafe fn vpkh(a: i16x8, b: i16x8) -> i8x16 {
        let a: i8x16 = transmute(a);
        let b: i8x16 = transmute(b);
        simd_shuffle(a, b, const { ShuffleMask::<16>::pack() })
    }
    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpkf))]
    unsafe fn vpkf(a: i32x4, b: i32x4) -> i16x8 {
        let a: i16x8 = transmute(a);
        let b: i16x8 = transmute(b);
        simd_shuffle(a, b, const { ShuffleMask::<8>::pack() })
    }
    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpkg))]
    unsafe fn vpkg(a: i64x2, b: i64x2) -> i32x4 {
        let a: i32x4 = transmute(a);
        let b: i32x4 = transmute(b);
        simd_shuffle(a, b, const { ShuffleMask::<4>::pack() })
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorPack<Other> {
        type Result;
        unsafe fn vec_pack(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorPack vec_pack]+ vpkh (vector_signed_short, vector_signed_short) -> vector_signed_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkh (vector_bool_short, vector_bool_short) -> vector_bool_char }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkf (vector_signed_int, vector_signed_int) -> vector_signed_short }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkf (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkf (vector_bool_int, vector_bool_int) -> vector_bool_short }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkg (vector_signed_long_long, vector_signed_long_long) -> vector_signed_int }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkg (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_int }
    impl_vec_trait! { [VectorPack vec_pack]+ vpkg (vector_bool_long_long, vector_bool_long_long) -> vector_bool_int }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPacks<Other> {
        type Result;
        unsafe fn vec_packs(self, b: Other) -> Self::Result;
    }

    // FIXME(llvm): https://github.com/llvm/llvm-project/issues/153655
    // Other targets can use a min/max for the saturation + a truncation.

    impl_vec_trait! { [VectorPacks vec_packs] vpksh (vector_signed_short, vector_signed_short) -> vector_signed_char }
    impl_vec_trait! { [VectorPacks vec_packs] vpklsh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacks vec_packs] vpksf (vector_signed_int, vector_signed_int) -> vector_signed_short }
    impl_vec_trait! { [VectorPacks vec_packs] vpklsf (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPacks vec_packs] vpksg (vector_signed_long_long, vector_signed_long_long) -> vector_signed_int }
    impl_vec_trait! { [VectorPacks vec_packs] vpklsg (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_int }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPacksu<Other> {
        type Result;
        unsafe fn vec_packsu(self, b: Other) -> Self::Result;
    }

    unsafe fn simd_smax<T: Copy>(a: T, b: T) -> T {
        simd_select::<T, T>(simd_gt::<T, T>(a, b), a, b)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpklsh))]
    unsafe fn vpacksuh(a: vector_signed_short, b: vector_signed_short) -> vector_unsigned_char {
        vpklsh(
            simd_smax(a, vector_signed_short([0; 8])),
            simd_smax(b, vector_signed_short([0; 8])),
        )
    }
    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpklsf))]
    unsafe fn vpacksuf(a: vector_signed_int, b: vector_signed_int) -> vector_unsigned_short {
        vpklsf(
            simd_smax(a, vector_signed_int([0; 4])),
            simd_smax(b, vector_signed_int([0; 4])),
        )
    }
    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vpklsg))]
    unsafe fn vpacksug(
        a: vector_signed_long_long,
        b: vector_signed_long_long,
    ) -> vector_unsigned_int {
        vpklsg(
            simd_smax(a, vector_signed_long_long([0; 2])),
            simd_smax(b, vector_signed_long_long([0; 2])),
        )
    }

    impl_vec_trait! { [VectorPacksu vec_packsu] vpacksuh (vector_signed_short, vector_signed_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacksu vec_packsu] vpklsh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_char }
    impl_vec_trait! { [VectorPacksu vec_packsu] vpacksuf (vector_signed_int, vector_signed_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPacksu vec_packsu] vpklsf (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_short }
    impl_vec_trait! { [VectorPacksu vec_packsu] vpacksug (vector_signed_long_long, vector_signed_long_long) -> vector_unsigned_int }
    impl_vec_trait! { [VectorPacksu vec_packsu] vpklsg (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_int }

    macro_rules! impl_vector_packs_cc {
        ($($intr:ident $ty:ident $outty:ident)*) => {
            $(
                #[inline]
                #[target_feature(enable = "vector")]
                #[cfg_attr(test, assert_instr($intr))]
                unsafe fn $intr(
                    a: $ty,
                    b: $ty,
                ) -> ($outty, i32) {
                    let PackedTuple { x, y } = super::$intr(a, b);
                    (x, y)
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorPacksCC for $ty {
                    type Result = $outty;

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_packs_cc(self, b: Self) -> (Self::Result, i32) {
                        $intr(self, b)
                    }
                }
            )*
        }
    }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPacksCC {
        type Result;
        unsafe fn vec_packs_cc(self, b: Self) -> (Self::Result, i32);
    }

    impl_vector_packs_cc! {
        vpkshs vector_signed_short vector_signed_char
        vpklshs vector_unsigned_short vector_unsigned_char
        vpksfs vector_signed_int vector_signed_short
        vpklsfs vector_unsigned_int vector_unsigned_short
        vpksgs vector_signed_long_long vector_signed_int
        vpklsgs vector_unsigned_long_long vector_unsigned_int
    }

    macro_rules! impl_vector_packsu_cc {
        ($($intr:ident $ty:ident $outty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorPacksuCC for $ty {
                    type Result = $outty;

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_packsu_cc(self, b: Self) -> (Self::Result, i32) {
                        $intr(self, b)
                    }
                }
            )*
        }
    }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPacksuCC {
        type Result;
        unsafe fn vec_packsu_cc(self, b: Self) -> (Self::Result, i32);
    }

    impl_vector_packsu_cc! {
        vpklshs vector_unsigned_short vector_unsigned_char
        vpklsfs vector_unsigned_int vector_unsigned_short
        vpklsgs vector_unsigned_long_long vector_unsigned_int
    }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorMadd {
        unsafe fn vec_madd(self, b: Self, c: Self) -> Self;
        unsafe fn vec_msub(self, b: Self, c: Self) -> Self;
    }

    test_impl! { vfmasb (a: vector_float, b: vector_float, c: vector_float) -> vector_float [simd_fma, "vector-enhancements-1" vfmasb] }
    test_impl! { vfmadb (a: vector_double, b: vector_double, c: vector_double) -> vector_double [simd_fma, vfmadb] }

    #[inline]
    unsafe fn simd_fms<T>(a: T, b: T, c: T) -> T {
        simd_fma(a, b, simd_neg(c))
    }

    test_impl! { vfmssb (a: vector_float, b: vector_float, c: vector_float) -> vector_float [simd_fms, "vector-enhancements-1" vfmssb] }
    test_impl! { vfmsdb (a: vector_double, b: vector_double, c: vector_double) -> vector_double [simd_fms, vfmsdb] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorMadd for vector_float {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_madd(self, b: Self, c: Self) -> Self {
            vfmasb(self, b, c)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_msub(self, b: Self, c: Self) -> Self {
            vfmssb(self, b, c)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorMadd for vector_double {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_madd(self, b: Self, c: Self) -> Self {
            vfmadb(self, b, c)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_msub(self, b: Self, c: Self) -> Self {
            vfmsdb(self, b, c)
        }
    }

    macro_rules! impl_vec_unpack {
        ($mask:ident $instr:ident $src:ident $shuffled:ident $dst:ident $width:literal) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn $instr(a: $src) -> $dst {
                simd_as(simd_shuffle::<_, _, $shuffled>(
                    a,
                    a,
                    const { ShuffleMask::<$width>::$mask() },
                ))
            }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorUnpackh {
        type Result;
        unsafe fn vec_unpackh(self) -> Self::Result;
    }

    impl_vec_unpack!(unpack_high vuphb vector_signed_char i8x8 vector_signed_short 8);
    impl_vec_unpack!(unpack_high vuphh vector_signed_short i16x4 vector_signed_int 4);
    impl_vec_unpack!(unpack_high vuphf vector_signed_int i32x2 vector_signed_long_long 2);

    impl_vec_unpack!(unpack_high vuplhb vector_unsigned_char u8x8 vector_unsigned_short 8);
    impl_vec_unpack!(unpack_high vuplhh vector_unsigned_short u16x4 vector_unsigned_int 4);
    impl_vec_unpack!(unpack_high vuplhf vector_unsigned_int u32x2 vector_unsigned_long_long 2);

    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuphb (vector_signed_char) -> vector_signed_short}
    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuphh (vector_signed_short) -> vector_signed_int}
    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuphf (vector_signed_int) -> vector_signed_long_long}

    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuplhb (vector_unsigned_char) -> vector_unsigned_short}
    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuplhh (vector_unsigned_short) -> vector_unsigned_int}
    impl_vec_trait! {[VectorUnpackh vec_unpackh] vuplhf (vector_unsigned_int) -> vector_unsigned_long_long}

    impl_vec_trait! {[VectorUnpackh vec_unpackh]+ vuplhb (vector_bool_char) -> vector_bool_short}
    impl_vec_trait! {[VectorUnpackh vec_unpackh]+ vuplhh (vector_bool_short) -> vector_bool_int}
    impl_vec_trait! {[VectorUnpackh vec_unpackh]+ vuplhf (vector_bool_int) -> vector_bool_long_long}

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorUnpackl {
        type Result;
        unsafe fn vec_unpackl(self) -> Self::Result;
    }

    // NOTE: `vuplh` is used for "unpack logical high", hence `vuplhw`.
    impl_vec_unpack!(unpack_low vuplb vector_signed_char i8x8 vector_signed_short 8);
    impl_vec_unpack!(unpack_low vuplhw vector_signed_short i16x4 vector_signed_int 4);
    impl_vec_unpack!(unpack_low vuplf vector_signed_int i32x2 vector_signed_long_long 2);

    impl_vec_unpack!(unpack_low vupllb vector_unsigned_char u8x8 vector_unsigned_short 8);
    impl_vec_unpack!(unpack_low vupllh vector_unsigned_short u16x4 vector_unsigned_int 4);
    impl_vec_unpack!(unpack_low vupllf vector_unsigned_int u32x2 vector_unsigned_long_long 2);

    impl_vec_trait! {[VectorUnpackl vec_unpackl] vuplb (vector_signed_char) -> vector_signed_short}
    impl_vec_trait! {[VectorUnpackl vec_unpackl] vuplhw (vector_signed_short) -> vector_signed_int}
    impl_vec_trait! {[VectorUnpackl vec_unpackl] vuplf (vector_signed_int) -> vector_signed_long_long}

    impl_vec_trait! {[VectorUnpackl vec_unpackl] vupllb (vector_unsigned_char) -> vector_unsigned_short}
    impl_vec_trait! {[VectorUnpackl vec_unpackl] vupllh (vector_unsigned_short) -> vector_unsigned_int}
    impl_vec_trait! {[VectorUnpackl vec_unpackl] vupllf (vector_unsigned_int) -> vector_unsigned_long_long}

    impl_vec_trait! {[VectorUnpackl vec_unpackl]+ vupllb (vector_bool_char) -> vector_bool_short}
    impl_vec_trait! {[VectorUnpackl vec_unpackl]+ vupllh (vector_bool_short) -> vector_bool_int}
    impl_vec_trait! {[VectorUnpackl vec_unpackl]+ vupllf (vector_bool_int) -> vector_bool_long_long}

    test_impl! { vec_vavgb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ vavgb, vavgb ] }
    test_impl! { vec_vavgh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [ vavgh, vavgh ] }
    test_impl! { vec_vavgf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [ vavgf, vavgf ] }
    test_impl! { vec_vavgg(a: vector_signed_long_long, b: vector_signed_long_long) -> vector_signed_long_long [ vavgg, vavgg ] }

    test_impl! { vec_vavglb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [ vavglb, vavglb ] }
    test_impl! { vec_vavglh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [ vavglh, vavglh ] }
    test_impl! { vec_vavglf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [ vavglf, vavglf ] }
    test_impl! { vec_vavglg(a: vector_unsigned_long_long, b: vector_unsigned_long_long) -> vector_unsigned_long_long [ vavglg, vavglg ] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorAvg<Other> {
        type Result;
        unsafe fn vec_avg(self, b: Other) -> Self::Result;
    }

    impl_vec_trait! { [VectorAvg vec_avg] 2 (vec_vavglb, vec_vavgb, vec_vavglh, vec_vavgh, vec_vavglf, vec_vavgf, vec_vavglg, vec_vavgg) }

    macro_rules! impl_mul {
        ([$Trait:ident $m:ident] $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl $Trait<$r> for $a {
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn $m(self, b: $b) -> $r {
                    $fun(transmute(self), transmute(b))
                }
            }
        };
        ([$Trait:ident $m:ident] $fun:ident ($a:ty, $b:ty, $c:ty) -> $r:ty) => {
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            impl $Trait for $a {
                type Result = $r;
                #[inline]
                #[target_feature(enable = "vector")]
                unsafe fn $m(self, b: $b, c: $c) -> $r {
                    $fun(self, b, c)
                }
            }
        };
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMule<Result> {
        unsafe fn vec_mule(self, b: Self) -> Result;
    }

    macro_rules! impl_vec_mul_even_odd {
        ($mask:ident $instr:ident $src:ident $shuffled:ident $dst:ident $width:literal) => {
            #[inline]
            #[target_feature(enable = "vector")]
            #[cfg_attr(test, assert_instr($instr))]
            unsafe fn $instr(a: $src, b: $src) -> $dst {
                let elems_a: $dst = simd_as(simd_shuffle::<_, _, $shuffled>(
                    a,
                    a, // this argument is ignored entirely.
                    const { ShuffleMask::<$width>::$mask() },
                ));

                let elems_b: $dst = simd_as(simd_shuffle::<_, _, $shuffled>(
                    b,
                    b, // this argument is ignored entirely.
                    const { ShuffleMask::<$width>::$mask() },
                ));

                simd_mul(elems_a, elems_b)
            }
        };
    }

    impl_vec_mul_even_odd! { even vmeb vector_signed_char i8x8 vector_signed_short 8 }
    impl_vec_mul_even_odd! { even vmeh vector_signed_short i16x4 vector_signed_int 4 }
    impl_vec_mul_even_odd! { even vmef vector_signed_int i32x2 vector_signed_long_long 2 }

    impl_vec_mul_even_odd! { even vmleb vector_unsigned_char u8x8 vector_unsigned_short 8 }
    impl_vec_mul_even_odd! { even vmleh vector_unsigned_short u16x4 vector_unsigned_int 4 }
    impl_vec_mul_even_odd! { even vmlef vector_unsigned_int u32x2 vector_unsigned_long_long 2 }

    impl_mul!([VectorMule vec_mule] vmeb (vector_signed_char, vector_signed_char) -> vector_signed_short );
    impl_mul!([VectorMule vec_mule] vmeh (vector_signed_short, vector_signed_short) -> vector_signed_int);
    impl_mul!([VectorMule vec_mule] vmef (vector_signed_int, vector_signed_int) -> vector_signed_long_long );

    impl_mul!([VectorMule vec_mule] vmleb (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_short );
    impl_mul!([VectorMule vec_mule] vmleh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_int);
    impl_mul!([VectorMule vec_mule] vmlef (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_long_long );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMulo<Result> {
        unsafe fn vec_mulo(self, b: Self) -> Result;
    }

    impl_vec_mul_even_odd! { odd vmob vector_signed_char i8x8 vector_signed_short 8 }
    impl_vec_mul_even_odd! { odd vmoh vector_signed_short i16x4 vector_signed_int 4 }
    impl_vec_mul_even_odd! { odd vmof vector_signed_int i32x2 vector_signed_long_long 2 }

    impl_vec_mul_even_odd! { odd vmlob vector_unsigned_char u8x8 vector_unsigned_short 8 }
    impl_vec_mul_even_odd! { odd vmloh vector_unsigned_short u16x4 vector_unsigned_int 4 }
    impl_vec_mul_even_odd! { odd vmlof vector_unsigned_int u32x2 vector_unsigned_long_long 2 }

    impl_mul!([VectorMulo vec_mulo] vmob (vector_signed_char, vector_signed_char) -> vector_signed_short );
    impl_mul!([VectorMulo vec_mulo] vmoh (vector_signed_short, vector_signed_short) -> vector_signed_int);
    impl_mul!([VectorMulo vec_mulo] vmof (vector_signed_int, vector_signed_int) -> vector_signed_long_long );

    impl_mul!([VectorMulo vec_mulo] vmlob (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_short );
    impl_mul!([VectorMulo vec_mulo] vmloh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_int);
    impl_mul!([VectorMulo vec_mulo] vmlof (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_long_long );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMulh<Result> {
        unsafe fn vec_mulh(self, b: Self) -> Result;
    }

    test_impl! { vec_vmhb(a: vector_signed_char, b: vector_signed_char) -> vector_signed_char [ vmhb, vmhb ] }
    test_impl! { vec_vmhh(a: vector_signed_short, b: vector_signed_short) -> vector_signed_short [ vmhh, vmhh ] }
    test_impl! { vec_vmhf(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int [ vmhf, vmhf ] }

    test_impl! { vec_vmlhb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_char [ vmlhb, vmlhb ] }
    test_impl! { vec_vmlhh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_short [ vmlhh, vmlhh ] }
    test_impl! { vec_vmlhf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int [ vmlhf, vmlhf ] }

    impl_mul!([VectorMulh vec_mulh] vec_vmhb (vector_signed_char, vector_signed_char) -> vector_signed_char);
    impl_mul!([VectorMulh vec_mulh] vec_vmhh (vector_signed_short, vector_signed_short) -> vector_signed_short);
    impl_mul!([VectorMulh vec_mulh] vec_vmhf (vector_signed_int, vector_signed_int) -> vector_signed_int);

    impl_mul!([VectorMulh vec_mulh] vec_vmlhb (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char);
    impl_mul!([VectorMulh vec_mulh] vec_vmlhh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short);
    impl_mul!([VectorMulh vec_mulh] vec_vmlhf (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int);

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMeadd {
        type Result;
        unsafe fn vec_meadd(self, b: Self, c: Self::Result) -> Self::Result;
    }

    test_impl! { vec_vmaeb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_short) -> vector_signed_short [ vmaeb, vmaeb ] }
    test_impl! { vec_vmaeh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_int) -> vector_signed_int[ vmaeh, vmaeh ] }
    test_impl! { vec_vmaef(a: vector_signed_int, b: vector_signed_int, c: vector_signed_long_long) -> vector_signed_long_long [ vmaef, vmaef ] }

    test_impl! { vec_vmaleb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short [ vmaleb, vmaleb ] }
    test_impl! { vec_vmaleh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int[ vmaleh, vmaleh ] }
    test_impl! { vec_vmalef(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long [ vmalef, vmalef ] }

    impl_mul!([VectorMeadd vec_meadd] vec_vmaeb (vector_signed_char, vector_signed_char, vector_signed_short) -> vector_signed_short );
    impl_mul!([VectorMeadd vec_meadd] vec_vmaeh (vector_signed_short, vector_signed_short, vector_signed_int) -> vector_signed_int);
    impl_mul!([VectorMeadd vec_meadd] vec_vmaef (vector_signed_int, vector_signed_int, vector_signed_long_long) -> vector_signed_long_long );

    impl_mul!([VectorMeadd vec_meadd] vec_vmaleb (vector_unsigned_char, vector_unsigned_char, vector_unsigned_short) -> vector_unsigned_short );
    impl_mul!([VectorMeadd vec_meadd] vec_vmaleh (vector_unsigned_short, vector_unsigned_short, vector_unsigned_int) -> vector_unsigned_int);
    impl_mul!([VectorMeadd vec_meadd] vec_vmalef (vector_unsigned_int, vector_unsigned_int, vector_unsigned_long_long) -> vector_unsigned_long_long );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMoadd {
        type Result;
        unsafe fn vec_moadd(self, b: Self, c: Self::Result) -> Self::Result;
    }

    test_impl! { vec_vmaob(a: vector_signed_char, b: vector_signed_char, c: vector_signed_short) -> vector_signed_short [ vmaob, vmaob ] }
    test_impl! { vec_vmaoh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_int) -> vector_signed_int[ vmaoh, vmaoh ] }
    test_impl! { vec_vmaof(a: vector_signed_int, b: vector_signed_int, c: vector_signed_long_long) -> vector_signed_long_long [ vmaof, vmaof ] }

    test_impl! { vec_vmalob(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short [ vmalob, vmalob ] }
    test_impl! { vec_vmaloh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int[ vmaloh, vmaloh ] }
    test_impl! { vec_vmalof(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long [ vmalof, vmalof ] }

    impl_mul!([VectorMoadd vec_moadd] vec_vmaob (vector_signed_char, vector_signed_char, vector_signed_short) -> vector_signed_short );
    impl_mul!([VectorMoadd vec_moadd] vec_vmaoh (vector_signed_short, vector_signed_short, vector_signed_int) -> vector_signed_int);
    impl_mul!([VectorMoadd vec_moadd] vec_vmaof (vector_signed_int, vector_signed_int, vector_signed_long_long) -> vector_signed_long_long );

    impl_mul!([VectorMoadd vec_moadd] vec_vmalob (vector_unsigned_char, vector_unsigned_char, vector_unsigned_short) -> vector_unsigned_short );
    impl_mul!([VectorMoadd vec_moadd] vec_vmaloh (vector_unsigned_short, vector_unsigned_short, vector_unsigned_int) -> vector_unsigned_int);
    impl_mul!([VectorMoadd vec_moadd] vec_vmalof (vector_unsigned_int, vector_unsigned_int, vector_unsigned_long_long) -> vector_unsigned_long_long );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMhadd {
        type Result;
        unsafe fn vec_mhadd(self, b: Self, c: Self::Result) -> Self::Result;
    }

    test_impl! { vec_vmahb(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char) -> vector_signed_char [ vmahb, vmahb ] }
    test_impl! { vec_vmahh(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short) -> vector_signed_short[ vmahh, vmahh ] }
    test_impl! { vec_vmahf(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int) -> vector_signed_int [ vmahf, vmahf ] }

    test_impl! { vec_vmalhb(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> vector_unsigned_char [ vmalhb, vmalhb ] }
    test_impl! { vec_vmalhh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short) -> vector_unsigned_short[ vmalhh, vmalhh ] }
    test_impl! { vec_vmalhf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int) -> vector_unsigned_int [ vmalhf, vmalhf ] }

    impl_mul!([VectorMhadd vec_mhadd] vec_vmahb (vector_signed_char, vector_signed_char, vector_signed_char) -> vector_signed_char );
    impl_mul!([VectorMhadd vec_mhadd] vec_vmahh (vector_signed_short, vector_signed_short, vector_signed_short) -> vector_signed_short);
    impl_mul!([VectorMhadd vec_mhadd] vec_vmahf (vector_signed_int, vector_signed_int, vector_signed_int) -> vector_signed_int );

    impl_mul!([VectorMhadd vec_mhadd] vec_vmalhb (vector_unsigned_char, vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char );
    impl_mul!([VectorMhadd vec_mhadd] vec_vmalhh (vector_unsigned_short, vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short);
    impl_mul!([VectorMhadd vec_mhadd] vec_vmalhf (vector_unsigned_int, vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorMladd {
        type Result;
        unsafe fn vec_mladd(self, b: Self, c: Self::Result) -> Self::Result;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    unsafe fn simd_mladd<T>(a: T, b: T, c: T) -> T {
        simd_add(simd_mul(a, b), c)
    }

    test_impl! { vec_vmal_ib(a: vector_signed_char, b: vector_signed_char, c: vector_signed_char) -> vector_signed_char [simd_mladd, vmalb ] }
    test_impl! { vec_vmal_ih(a: vector_signed_short, b: vector_signed_short, c: vector_signed_short) -> vector_signed_short[simd_mladd, vmalhw ] }
    test_impl! { vec_vmal_if(a: vector_signed_int, b: vector_signed_int, c: vector_signed_int) -> vector_signed_int [simd_mladd, vmalf ] }

    test_impl! { vec_vmal_ub(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_char) -> vector_unsigned_char [simd_mladd, vmalb ] }
    test_impl! { vec_vmal_uh(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_short) -> vector_unsigned_short[simd_mladd, vmalhw ] }
    test_impl! { vec_vmal_uf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_int) -> vector_unsigned_int [simd_mladd, vmalf ] }

    impl_mul!([VectorMladd vec_mladd] vec_vmal_ib (vector_signed_char, vector_signed_char, vector_signed_char) -> vector_signed_char );
    impl_mul!([VectorMladd vec_mladd] vec_vmal_ih (vector_signed_short, vector_signed_short, vector_signed_short) -> vector_signed_short);
    impl_mul!([VectorMladd vec_mladd] vec_vmal_if (vector_signed_int, vector_signed_int, vector_signed_int) -> vector_signed_int );

    impl_mul!([VectorMladd vec_mladd] vec_vmal_ub (vector_unsigned_char, vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char );
    impl_mul!([VectorMladd vec_mladd] vec_vmal_uh (vector_unsigned_short, vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short);
    impl_mul!([VectorMladd vec_mladd] vec_vmal_uf (vector_unsigned_int, vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorGfmsum<Result> {
        unsafe fn vec_gfmsum(self, b: Self) -> Result;
    }

    test_impl! { vec_vgfmb(a: vector_unsigned_char, b: vector_unsigned_char) -> vector_unsigned_short [ vgfmb, vgfmb ] }
    test_impl! { vec_vgfmh(a: vector_unsigned_short, b: vector_unsigned_short) -> vector_unsigned_int[ vgfmh, vgfmh] }
    test_impl! { vec_vgfmf(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_long_long [ vgfmf, vgfmf ] }

    impl_mul!([VectorGfmsum vec_gfmsum] vec_vgfmb (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_short );
    impl_mul!([VectorGfmsum vec_gfmsum] vec_vgfmh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_int);
    impl_mul!([VectorGfmsum vec_gfmsum] vec_vgfmf (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_long_long );

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorGfmsumAccum {
        type Result;
        unsafe fn vec_gfmsum_accum(self, b: Self, c: Self::Result) -> Self::Result;
    }

    test_impl! { vec_vgfmab(a: vector_unsigned_char, b: vector_unsigned_char, c: vector_unsigned_short) -> vector_unsigned_short [ vgfmab, vgfmab ] }
    test_impl! { vec_vgfmah(a: vector_unsigned_short, b: vector_unsigned_short, c: vector_unsigned_int) -> vector_unsigned_int[ vgfmah, vgfmah] }
    test_impl! { vec_vgfmaf(a: vector_unsigned_int, b: vector_unsigned_int, c: vector_unsigned_long_long) -> vector_unsigned_long_long [ vgfmaf, vgfmaf ] }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorGfmsumAccum for vector_unsigned_char {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_gfmsum_accum(self, b: Self, c: Self::Result) -> Self::Result {
            vec_vgfmab(self, b, c)
        }
    }
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorGfmsumAccum for vector_unsigned_short {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_gfmsum_accum(self, b: Self, c: Self::Result) -> Self::Result {
            vec_vgfmah(self, b, c)
        }
    }
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorGfmsumAccum for vector_unsigned_int {
        type Result = vector_unsigned_long_long;
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_gfmsum_accum(self, b: Self, c: Self::Result) -> Self::Result {
            vec_vgfmaf(self, b, c)
        }
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vgef, D = 3))]
    unsafe fn vgef<const D: u32>(
        a: vector_unsigned_int,
        b: vector_unsigned_int,
        c: *const u32,
    ) -> vector_unsigned_int {
        static_assert_uimm_bits!(D, 2);
        let offset: u32 = simd_extract(b, D);
        let ptr = c.byte_add(offset as usize);
        let value = ptr.read();
        simd_insert(a, D, value)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vgeg, D = 1))]
    unsafe fn vgeg<const D: u32>(
        a: vector_unsigned_long_long,
        b: vector_unsigned_long_long,
        c: *const u64,
    ) -> vector_unsigned_long_long {
        static_assert_uimm_bits!(D, 1);
        let offset: u64 = simd_extract(b, D);
        let ptr = c.byte_add(offset as usize);
        let value = ptr.read();
        simd_insert(a, D, value)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorGatherElement {
        type Element;
        type Offset;
        unsafe fn vec_gather_element<const D: u32>(
            self,
            b: Self::Offset,
            c: *const Self::Element,
        ) -> Self;
    }

    macro_rules! impl_vec_gather_element {
        ($($instr:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorGatherElement for $ty {
                    type Element = l_t_t!($ty);
                    type Offset = t_u!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_gather_element<const D: u32>(self, b: Self::Offset, c: *const Self::Element) -> Self {
                        transmute($instr::<D>(transmute(self), b, c.cast()))
                    }
                }
            )*
        }
    }

    impl_vec_gather_element! {
        vgef vector_signed_int
        vgef vector_bool_int
        vgef vector_unsigned_int

        vgeg vector_signed_long_long
        vgeg vector_bool_long_long
        vgeg vector_unsigned_long_long

        vgef vector_float
        vgeg vector_double
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vscef, D = 3))]
    unsafe fn vscef<const D: u32>(a: vector_unsigned_int, b: vector_unsigned_int, c: *mut u32) {
        static_assert_uimm_bits!(D, 2);
        let value = simd_extract(a, D);
        let offset: u32 = simd_extract(b, D);
        let ptr = c.byte_add(offset as usize);
        ptr.write(value);
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsceg, D = 1))]
    unsafe fn vsceg<const D: u32>(
        a: vector_unsigned_long_long,
        b: vector_unsigned_long_long,
        c: *mut u64,
    ) {
        static_assert_uimm_bits!(D, 1);
        let value = simd_extract(a, D);
        let offset: u64 = simd_extract(b, D);
        let ptr = c.byte_add(offset as usize);
        ptr.write(value);
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorScatterElement {
        type Element;
        type Offset;
        unsafe fn vec_scatter_element<const D: u32>(self, b: Self::Offset, c: *mut Self::Element);
    }

    macro_rules! impl_vec_scatter_element {
        ($($instr:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorScatterElement for $ty {
                    type Element = l_t_t!($ty);
                    type Offset = t_u!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_scatter_element<const D: u32>(self, b: Self::Offset, c: *mut Self::Element) {
                        $instr::<D>(transmute(self), b, c.cast())
                    }
                }
            )*
        }
    }

    impl_vec_scatter_element! {
        vscef vector_signed_int
        vscef vector_bool_int
        vscef vector_unsigned_int

        vsceg vector_signed_long_long
        vsceg vector_bool_long_long
        vsceg vector_unsigned_long_long

        vscef vector_float
        vsceg vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSel<Mask>: Sized {
        unsafe fn vec_sel(self, b: Self, c: Mask) -> Self;
    }

    macro_rules! impl_vec_sel {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSel<t_u!($ty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_sel(self, b: Self, c: t_u!($ty)) -> Self {
                        let b = simd_and(transmute(b), c);
                        let a = simd_and(transmute(self), simd_xor(c, transmute(vector_signed_char([!0; 16]))));
                        transmute(simd_or(a, b))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSel<t_b!($ty)> for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_sel(self, b: Self, c: t_b!($ty)) -> Self {
                        // defer to the implementation with an unsigned mask
                        self.vec_sel(b, transmute::<_, t_u!($ty)>(c))
                    }
                }
            )*
        }
    }

    impl_vec_sel! {
        vector_signed_char
        vector_signed_short
        vector_signed_int
        vector_signed_long_long

        vector_unsigned_char
        vector_unsigned_short
        vector_unsigned_int
        vector_unsigned_long_long

        vector_bool_char
        vector_bool_short
        vector_bool_int
        vector_bool_long_long

        vector_float
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFpTestDataClass {
        type Result;
        unsafe fn vec_fp_test_data_class<const CLASS: u32>(self) -> (Self::Result, i32);
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorFpTestDataClass for vector_float {
        type Result = vector_bool_int;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_fp_test_data_class<const CLASS: u32>(self) -> (Self::Result, i32) {
            let PackedTuple { x, y } = vftcisb(self, CLASS);
            (x, y)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorFpTestDataClass for vector_double {
        type Result = vector_bool_long_long;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_fp_test_data_class<const CLASS: u32>(self) -> (Self::Result, i32) {
            let PackedTuple { x, y } = vftcidb(self, CLASS);
            (x, y)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCompare {
        unsafe fn vec_all_lt(self, other: Self) -> i32;
        unsafe fn vec_all_le(self, other: Self) -> i32;
        unsafe fn vec_all_gt(self, other: Self) -> i32;
        unsafe fn vec_all_ge(self, other: Self) -> i32;
    }

    // NOTE: this implementation is currently non-optimal, but it does work for floats even with
    // only `vector` enabled.
    //
    // - https://github.com/llvm/llvm-project/issues/129434
    // - https://github.com/llvm/llvm-project/issues/130424
    macro_rules! impl_vec_compare {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorCompare for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_all_lt(self, other: Self) -> i32 {
                        simd_reduce_all(simd_lt::<_, t_b!($ty)>(self, other)) as i32
                    }
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_all_le(self, other: Self) -> i32 {
                        simd_reduce_all(simd_le::<_, t_b!($ty)>(self, other)) as i32
                    }
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_all_gt(self, other: Self) -> i32 {
                        simd_reduce_all(simd_gt::<_, t_b!($ty)>(self, other)) as i32
                    }
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_all_ge(self, other: Self) -> i32 {
                        simd_reduce_all(simd_ge::<_, t_b!($ty)>(self, other)) as i32
                    }
                }
            )*
        }
    }

    impl_vec_compare! {
        vector_signed_char
        vector_unsigned_char

        vector_signed_short
        vector_unsigned_short

        vector_signed_int
        vector_unsigned_int
        vector_float

        vector_signed_long_long
        vector_unsigned_long_long
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorTestMask {
        type Mask;
        unsafe fn vec_test_mask(self, other: Self::Mask) -> i32;
    }

    macro_rules! impl_vec_test_mask {
        ($($instr:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorTestMask for $ty {
                    type Mask = t_u!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_test_mask(self, other: Self::Mask) -> i32 {
                        vtm(transmute(self), transmute(other))
                    }
                }
            )*
        }
    }

    impl_vec_test_mask! {
        vector_signed_char
        vector_signed_short
        vector_signed_int
        vector_signed_long_long

        vector_unsigned_char
        vector_unsigned_short
        vector_unsigned_int
        vector_unsigned_long_long

        vector_float
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSearchString {
        unsafe fn vec_search_string_cc(
            self,
            b: Self,
            c: vector_unsigned_char,
        ) -> (vector_unsigned_char, i32);

        unsafe fn vec_search_string_until_zero_cc(
            self,
            b: Self,
            c: vector_unsigned_char,
        ) -> (vector_unsigned_char, i32);
    }

    macro_rules! impl_vec_search_string{
        ($($intr_s:ident $intr_sz:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSearchString for $ty {
                    #[inline]
                    #[target_feature(enable = "vector-enhancements-2")]
                    unsafe fn vec_search_string_cc(self, b: Self, c: vector_unsigned_char) -> (vector_unsigned_char, i32) {
                        let PackedTuple { x,y } = $intr_s(transmute(self), transmute(b), c);
                        (x, y)
                    }

                    #[inline]
                    #[target_feature(enable = "vector-enhancements-2")]
                    unsafe fn vec_search_string_until_zero_cc(self, b: Self, c: vector_unsigned_char) -> (vector_unsigned_char, i32) {
                        let PackedTuple { x,y } = $intr_sz(transmute(self), transmute(b), c);
                        (x, y)
                    }
                }

            )*
        }
    }

    impl_vec_search_string! {
        vstrsb vstrszb vector_signed_char
        vstrsb vstrszb vector_bool_char
        vstrsb vstrszb vector_unsigned_char

        vstrsh vstrszh vector_signed_short
        vstrsh vstrszh vector_bool_short
        vstrsh vstrszh vector_unsigned_short

        vstrsf vstrszf vector_signed_int
        vstrsf vstrszf vector_bool_int
        vstrsf vstrszf vector_unsigned_int
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vcdgb))]
    pub unsafe fn vcdgb(a: vector_signed_long_long) -> vector_double {
        simd_as(a)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vcdlgb))]
    pub unsafe fn vcdlgb(a: vector_unsigned_long_long) -> vector_double {
        simd_as(a)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorDouble {
        unsafe fn vec_double(self) -> vector_double;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorDouble for vector_signed_long_long {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_double(self) -> vector_double {
            vcdgb(self)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorDouble for vector_unsigned_long_long {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_double(self) -> vector_double {
            vcdlgb(self)
        }
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vcefb)
    )]
    pub unsafe fn vcefb(a: vector_signed_int) -> vector_float {
        simd_as(a)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(
        all(test, target_feature = "vector-enhancements-2"),
        assert_instr(vcelfb)
    )]
    pub unsafe fn vcelfb(a: vector_unsigned_int) -> vector_float {
        simd_as(a)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorFloat {
        unsafe fn vec_float(self) -> vector_float;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorFloat for vector_signed_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_float(self) -> vector_float {
            vcefb(self)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorFloat for vector_unsigned_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_float(self) -> vector_float {
            vcelfb(self)
        }
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorExtendSigned64 {
        unsafe fn vec_extend_s64(self) -> vector_signed_long_long;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsegb))]
    pub unsafe fn vsegb(a: vector_signed_char) -> vector_signed_long_long {
        simd_as(simd_shuffle::<_, _, i8x2>(
            a,
            a,
            const { u32x2::from_array([7, 15]) },
        ))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsegh))]
    pub unsafe fn vsegh(a: vector_signed_short) -> vector_signed_long_long {
        simd_as(simd_shuffle::<_, _, i16x2>(
            a,
            a,
            const { u32x2::from_array([3, 7]) },
        ))
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsegf))]
    pub unsafe fn vsegf(a: vector_signed_int) -> vector_signed_long_long {
        simd_as(simd_shuffle::<_, _, i32x2>(
            a,
            a,
            const { u32x2::from_array([1, 3]) },
        ))
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorExtendSigned64 for vector_signed_char {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_extend_s64(self) -> vector_signed_long_long {
            vsegb(self)
        }
    }
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorExtendSigned64 for vector_signed_short {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_extend_s64(self) -> vector_signed_long_long {
            vsegh(self)
        }
    }
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    impl VectorExtendSigned64 for vector_signed_int {
        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_extend_s64(self) -> vector_signed_long_long {
            vsegf(self)
        }
    }

    // NOTE: VectorSigned and VectorUnsigned make strong safety assumptions around floats.
    // This is what C provides, but even IBM does not clearly document these constraints.
    //
    // https://doc.rust-lang.org/std/intrinsics/simd/fn.simd_cast.html

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSigned {
        type Result;
        unsafe fn vec_signed(self) -> Self::Result;
    }

    test_impl! { vcgsb (a: vector_float) -> vector_signed_int [simd_cast, "vector-enhancements-2" vcgsb] }
    test_impl! { vcgdb (a: vector_double) -> vector_signed_long_long [simd_cast, vcgdb] }

    impl_vec_trait! { [VectorSigned vec_signed] vcgsb (vector_float) -> vector_signed_int }
    impl_vec_trait! { [VectorSigned vec_signed] vcgdb (vector_double) -> vector_signed_long_long }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorUnsigned {
        type Result;
        unsafe fn vec_unsigned(self) -> Self::Result;
    }

    test_impl! { vclgsb (a: vector_float) -> vector_unsigned_int [simd_cast, "vector-enhancements-2" vclgsb] }
    test_impl! { vclgdb (a: vector_double) -> vector_unsigned_long_long [simd_cast, vclgdb] }

    impl_vec_trait! { [VectorUnsigned vec_unsigned] vclgsb (vector_float) -> vector_unsigned_int }
    impl_vec_trait! { [VectorUnsigned vec_unsigned] vclgdb (vector_double) -> vector_unsigned_long_long }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCopyUntilZero {
        unsafe fn vec_cp_until_zero(self) -> Self;
    }

    test_impl! { vec_vistrb (a: vector_unsigned_char) -> vector_unsigned_char [vistrb, vistrb] }
    test_impl! { vec_vistrh (a: vector_unsigned_short) -> vector_unsigned_short [vistrh, vistrh] }
    test_impl! { vec_vistrf (a: vector_unsigned_int) -> vector_unsigned_int [vistrf, vistrf] }

    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrb (vector_signed_char) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrb (vector_bool_char) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrb (vector_unsigned_char) }

    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrh (vector_signed_short) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrh (vector_bool_short) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrh (vector_unsigned_short) }

    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrf (vector_signed_int) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrf (vector_bool_int) }
    impl_vec_trait! { [VectorCopyUntilZero vec_cp_until_zero]+ vec_vistrf (vector_unsigned_int) }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCopyUntilZeroCC: Sized {
        unsafe fn vec_cp_until_zero_cc(self) -> (Self, i32);
    }

    test_impl! { vec_vistrbs (a: vector_unsigned_char) -> PackedTuple<vector_unsigned_char, i32> [vistrbs, vistrbs] }
    test_impl! { vec_vistrhs (a: vector_unsigned_short) -> PackedTuple<vector_unsigned_short, i32> [vistrhs, vistrhs] }
    test_impl! { vec_vistrfs (a: vector_unsigned_int) -> PackedTuple<vector_unsigned_int, i32> [vistrfs, vistrfs] }

    macro_rules! impl_vec_copy_until_zero_cc {
        ($($intr:ident $ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorCopyUntilZeroCC for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cp_until_zero_cc(self) -> (Self, i32) {
                        let PackedTuple { x,y } = $intr(transmute(self));
                        (transmute(x), y)
                    }
                }

            )*
        }
    }

    impl_vec_copy_until_zero_cc! {
        vec_vistrbs vector_signed_char
        vec_vistrbs vector_bool_char
        vec_vistrbs vector_unsigned_char

        vec_vistrhs vector_signed_short
        vec_vistrhs vector_bool_short
        vec_vistrhs vector_unsigned_short

        vec_vistrfs vector_signed_int
        vec_vistrfs vector_bool_int
        vec_vistrfs vector_unsigned_int
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSrdb {
        unsafe fn vec_srdb<const C: u32>(self, b: Self) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorSld {
        unsafe fn vec_sld<const C: u32>(self, b: Self) -> Self;

        unsafe fn vec_sldw<const C: u32>(self, b: Self) -> Self;

        unsafe fn vec_sldb<const C: u32>(self, b: Self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsldb))]
    unsafe fn test_vec_sld(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        a.vec_sld::<13>(b)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vsldb))]
    unsafe fn test_vec_sldw(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        a.vec_sldw::<3>(b)
    }

    #[inline]
    #[target_feature(enable = "vector-enhancements-2")]
    #[cfg_attr(test, assert_instr(vsld))]
    unsafe fn test_vec_sldb(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        a.vec_sldb::<7>(b)
    }

    #[inline]
    #[target_feature(enable = "vector-enhancements-2")]
    #[cfg_attr(test, assert_instr(vsrd))]
    unsafe fn test_vec_srdb(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        a.vec_srdb::<7>(b)
    }

    macro_rules! impl_vec_sld {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSld for $ty {
                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_sld<const C: u32>(self, b: Self) -> Self {
                        static_assert_uimm_bits!(C, 4);
                        transmute(u128::funnel_shl(transmute(self), transmute(b), C  * 8))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_sldw<const C: u32>(self, b: Self) -> Self {
                        static_assert_uimm_bits!(C, 2);
                        transmute(u128::funnel_shl(transmute(self), transmute(b), C * 4 * 8))
                    }

                    #[inline]
                    #[target_feature(enable = "vector-enhancements-2")]
                    unsafe fn vec_sldb<const C: u32>(self, b: Self) -> Self {
                        static_assert_uimm_bits!(C, 3);
                        transmute(u128::funnel_shl(transmute(self), transmute(b), C))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorSrdb for $ty {
                    #[inline]
                    #[target_feature(enable = "vector-enhancements-2")]
                    unsafe fn vec_srdb<const C: u32>(self, b: Self) -> Self {
                        static_assert_uimm_bits!(C, 3);
                        transmute(vsrd(transmute(self), transmute(b), C))
                        // FIXME(llvm): https://github.com/llvm/llvm-project/issues/129955#issuecomment-3207488190
                        // LLVM currently rewrites `fshr` to `fshl`, and the logic in the s390x
                        // backend cannot deal with that yet.
                        // #[link_name = "llvm.fshr.i128"] fn fshr_i128(a: u128, b: u128, c: u128) -> u128;
                        // transmute(fshr_i128(transmute(self), transmute(b), const { C as u128 }))
                    }
                }
            )*
        }
    }

    impl_vec_sld! {
        vector_signed_char
        vector_bool_char
        vector_unsigned_char

        vector_signed_short
        vector_bool_short
        vector_unsigned_short

        vector_signed_int
        vector_bool_int
        vector_unsigned_int

        vector_signed_long_long
        vector_bool_long_long
        vector_unsigned_long_long

        vector_float
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorCompareRange: Sized {
        type Result;

        unsafe fn vstrc<const IMM: u32>(self, b: Self, c: Self) -> Self::Result;
        unsafe fn vstrcz<const IMM: u32>(self, b: Self, c: Self) -> Self::Result;
        unsafe fn vstrcs<const IMM: u32>(self, b: Self, c: Self) -> (Self::Result, i32);
        unsafe fn vstrczs<const IMM: u32>(self, b: Self, c: Self) -> (Self::Result, i32);
    }

    const fn validate_compare_range_imm(imm: u32) {
        if !matches!(imm, 0 | 4 | 8 | 12) {
            panic!("IMM needs to be one of 0, 4, 8, 12");
        }
    }

    macro_rules! impl_compare_range {
        ($($ty:ident $vstrc:ident $vstrcs:ident $vstrcz:ident $vstrczs:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorCompareRange for $ty {
                    type Result = t_b!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vstrc<const IMM: u32>(self, b: Self, c: Self) -> Self::Result {
                        const { validate_compare_range_imm };
                        $vstrc(self, b, c, IMM)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vstrcz<const IMM: u32>(self, b: Self, c: Self) -> Self::Result {
                        const { validate_compare_range_imm };
                        $vstrcz(self, b, c, IMM)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vstrcs<const IMM: u32>(self, b: Self, c: Self) -> (Self::Result, i32) {
                        const { validate_compare_range_imm };
                        let PackedTuple { x, y } = $vstrcs(self, b, c, IMM);
                        (x,y)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vstrczs<const IMM: u32>(self, b: Self, c: Self) -> (Self::Result, i32) {
                        const { validate_compare_range_imm };
                        let PackedTuple { x, y } = $vstrczs(self, b, c, IMM);
                        (x,y)
                    }
                }
            )*
        }
    }

    impl_compare_range! {
        vector_unsigned_char    vstrcb vstrcbs vstrczb vstrczbs
        vector_unsigned_short   vstrch vstrchs vstrczh vstrczhs
        vector_unsigned_int     vstrcf vstrcfs vstrczf vstrczfs
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorComparePredicate: Sized {
        type Result;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmpgt(self, other: Self) -> Self::Result {
            simd_gt(self, other)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmpge(self, other: Self) -> Self::Result {
            simd_ge(self, other)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmplt(self, other: Self) -> Self::Result {
            simd_lt(self, other)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmple(self, other: Self) -> Self::Result {
            simd_le(self, other)
        }
    }

    macro_rules! impl_compare_predicate {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorComparePredicate for $ty {
                    type Result = t_b!($ty);
                }
            )*
        }
    }

    impl_compare_predicate! {
        vector_signed_char
        vector_unsigned_char

        vector_signed_short
        vector_unsigned_short

        vector_signed_int
        vector_unsigned_int
        vector_float

        vector_signed_long_long
        vector_unsigned_long_long
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorEquality: Sized {
        type Result;

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmpeq(self, other: Self) -> Self::Result {
            simd_eq(self, other)
        }

        #[inline]
        #[target_feature(enable = "vector")]
        unsafe fn vec_cmpne(self, other: Self) -> Self::Result {
            simd_ne(self, other)
        }
    }

    macro_rules! impl_compare_equality {
        ($($ty:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorEquality for $ty {
                    type Result = t_b!($ty);
                }
            )*
        }
    }

    impl_compare_equality! {
        vector_bool_char
        vector_signed_char
        vector_unsigned_char

        vector_bool_short
        vector_signed_short
        vector_unsigned_short

        vector_bool_int
        vector_signed_int
        vector_unsigned_int
        vector_float

        vector_bool_long_long
        vector_signed_long_long
        vector_unsigned_long_long
        vector_double
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorEqualityIdx: Sized {
        type Result;

        unsafe fn vec_cmpeq_idx(self, other: Self) -> Self::Result;
        unsafe fn vec_cmpne_idx(self, other: Self) -> Self::Result;

        unsafe fn vec_cmpeq_idx_cc(self, other: Self) -> (Self::Result, i32);
        unsafe fn vec_cmpne_idx_cc(self, other: Self) -> (Self::Result, i32);

        unsafe fn vec_cmpeq_or_0_idx(self, other: Self) -> Self::Result;
        unsafe fn vec_cmpne_or_0_idx(self, other: Self) -> Self::Result;

        unsafe fn vec_cmpeq_or_0_idx_cc(self, other: Self) -> (Self::Result, i32);
        unsafe fn vec_cmpne_or_0_idx_cc(self, other: Self) -> (Self::Result, i32);
    }

    macro_rules! impl_compare_equality_idx {
        ($($ty:ident $ret:ident
                $cmpeq:ident $cmpne:ident
                $cmpeq_or_0:ident $cmpne_or_0:ident
                $cmpeq_cc:ident $cmpne_cc:ident
                $cmpeq_or_0_cc:ident $cmpne_or_0_cc:ident
        )*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorEqualityIdx for $ty {
                    type Result = $ret;

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpeq_idx(self, other: Self) -> Self::Result {
                        transmute($cmpeq(transmute(self), transmute(other)))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpne_idx(self, other: Self) -> Self::Result {
                        transmute($cmpne(transmute(self), transmute(other)))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpeq_or_0_idx(self, other: Self) -> Self::Result {
                        transmute($cmpeq_or_0(transmute(self), transmute(other)))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpne_or_0_idx(self, other: Self) -> Self::Result {
                        transmute($cmpne_or_0(transmute(self), transmute(other)))
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpeq_idx_cc(self, other: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $cmpeq_cc(transmute(self), transmute(other));
                        (transmute(x), y)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpne_idx_cc(self, other: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $cmpne_cc(transmute(self), transmute(other));
                        (transmute(x),y)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpeq_or_0_idx_cc(self, other: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $cmpeq_or_0_cc(transmute(self), transmute(other));
                        (transmute(x), y)
                    }

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_cmpne_or_0_idx_cc(self, other: Self) -> (Self::Result, i32) {
                        let PackedTuple { x, y } = $cmpne_or_0_cc(transmute(self), transmute(other));
                        (transmute(x),y)
                    }
                }
            )*
        }
    }

    impl_compare_equality_idx! {
        vector_signed_char vector_signed_char               vfeeb vfeneb vfeezb vfenezb vfeebs vfenebs vfeezbs vfenezbs
        vector_bool_char vector_unsigned_char               vfeeb vfeneb vfeezb vfenezb vfeebs vfenebs vfeezbs vfenezbs
        vector_unsigned_char vector_unsigned_char           vfeeb vfeneb vfeezb vfenezb vfeebs vfenebs vfeezbs vfenezbs
        vector_signed_short vector_signed_short             vfeeh vfeneh vfeezh vfenezh vfeehs vfenehs vfeezhs vfenezhs
        vector_bool_short  vector_unsigned_short            vfeeh vfeneh vfeezh vfenezh vfeehs vfenehs vfeezhs vfenezhs
        vector_unsigned_short vector_unsigned_short         vfeeh vfeneh vfeezh vfenezh vfeehs vfenehs vfeezhs vfenezhs
        vector_signed_int vector_signed_int                 vfeef vfenef vfeezf vfenezf vfeefs vfenefs vfeezfs vfenezfs
        vector_bool_int  vector_unsigned_int                vfeef vfenef vfeezf vfenezf vfeefs vfenefs vfeezfs vfenezfs
        vector_unsigned_int vector_unsigned_int             vfeef vfenef vfeezf vfenezf vfeefs vfenefs vfeezfs vfenezfs
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorExtract {
        type ElementType;

        unsafe fn vec_extract(a: Self, b: i32) -> Self::ElementType;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlgvb))]
    unsafe fn vlgvb(a: vector_unsigned_char, b: i32) -> u8 {
        simd_extract_dyn(a, b as u32 % 16)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlgvh))]
    unsafe fn vlgvh(a: vector_unsigned_short, b: i32) -> u16 {
        simd_extract_dyn(a, b as u32 % 8)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlgvf))]
    unsafe fn vlgvf(a: vector_unsigned_int, b: i32) -> u32 {
        simd_extract_dyn(a, b as u32 % 4)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlgvg))]
    unsafe fn vlgvg(a: vector_unsigned_long_long, b: i32) -> u64 {
        simd_extract_dyn(a, b as u32 % 2)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorInsert {
        type ElementType;

        unsafe fn vec_insert(a: Self::ElementType, b: Self, c: i32) -> Self;
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorPromote: Sized {
        type ElementType;

        unsafe fn vec_promote(a: Self::ElementType, b: i32) -> MaybeUninit<Self>;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlvgb))]
    unsafe fn vlvgb(a: u8, b: vector_unsigned_char, c: i32) -> vector_unsigned_char {
        simd_insert_dyn(b, c as u32 % 16, a)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlvgh))]
    unsafe fn vlvgh(a: u16, b: vector_unsigned_short, c: i32) -> vector_unsigned_short {
        simd_insert_dyn(b, c as u32 % 8, a)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlvgf))]
    unsafe fn vlvgf(a: u32, b: vector_unsigned_int, c: i32) -> vector_unsigned_int {
        simd_insert_dyn(b, c as u32 % 4, a)
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vlvgg))]
    unsafe fn vlvgg(a: u64, b: vector_unsigned_long_long, c: i32) -> vector_unsigned_long_long {
        simd_insert_dyn(b, c as u32 % 2, a)
    }

    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub trait VectorInsertAndZero {
        type ElementType;

        unsafe fn vec_insert_and_zero(a: *const Self::ElementType) -> Self;
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vllezb))]
    unsafe fn vllezb(x: *const u8) -> vector_unsigned_char {
        vector_unsigned_char([0, 0, 0, 0, 0, 0, 0, *x, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vllezh))]
    unsafe fn vllezh(x: *const u16) -> vector_unsigned_short {
        vector_unsigned_short([0, 0, 0, *x, 0, 0, 0, 0])
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vllezf))]
    unsafe fn vllezf(x: *const u32) -> vector_unsigned_int {
        vector_unsigned_int([0, *x, 0, 0])
    }

    #[inline]
    #[target_feature(enable = "vector")]
    #[cfg_attr(test, assert_instr(vllezg))]
    unsafe fn vllezg(x: *const u64) -> vector_unsigned_long_long {
        vector_unsigned_long_long([*x, 0])
    }

    macro_rules! impl_extract_insert {
        ($($ty:ident $extract_intr:ident $insert_intr:ident $insert_and_zero_intr:ident)*) => {
            $(
                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorExtract for $ty {
                    type ElementType = l_t_t!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_extract(a: Self, b: i32) -> Self::ElementType {
                        transmute($extract_intr(transmute(a), b))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorInsert for $ty {
                    type ElementType = l_t_t!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_insert(a: Self::ElementType, b: Self, c: i32) -> Self {
                        transmute($insert_intr(transmute(a), transmute(b), c))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorInsertAndZero for $ty {
                    type ElementType = l_t_t!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_insert_and_zero(a: *const Self::ElementType) -> Self {
                        transmute($insert_and_zero_intr(a.cast()))
                    }
                }

                #[unstable(feature = "stdarch_s390x", issue = "135681")]
                impl VectorPromote for $ty {
                    type ElementType = l_t_t!($ty);

                    #[inline]
                    #[target_feature(enable = "vector")]
                    unsafe fn vec_promote(a: Self::ElementType, c: i32) -> MaybeUninit<Self> {
                        // Rust does not currently support `MaybeUninit` element types to simd
                        // vectors. In C/LLVM that is allowed (using poison values). So rust will
                        // use an extra instruction to zero the memory.
                        let b = MaybeUninit::<$ty>::zeroed();
                        MaybeUninit::new(transmute($insert_intr(transmute(a), transmute(b), c)))
                    }
                }
            )*
        }

    }

    impl_extract_insert! {
        vector_signed_char          vlgvb vlvgb vllezb
        vector_unsigned_char        vlgvb vlvgb vllezb
        vector_signed_short         vlgvh vlvgh vllezh
        vector_unsigned_short       vlgvh vlvgh vllezh
        vector_signed_int           vlgvf vlvgf vllezf
        vector_unsigned_int         vlgvf vlvgf vllezf
        vector_signed_long_long     vlgvg vlvgg vllezg
        vector_unsigned_long_long   vlgvg vlvgg vllezg
        vector_float                vlgvf vlvgf vllezf
        vector_double               vlgvg vlvgg vllezg
    }
}

/// Load Count to Block Boundary
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(lcbb, BLOCK_BOUNDARY = 512))]
unsafe fn __lcbb<const BLOCK_BOUNDARY: u16>(ptr: *const u8) -> u32 {
    lcbb(ptr, const { validate_block_boundary(BLOCK_BOUNDARY) })
}

/// Vector Add
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_add<T: sealed::VectorAdd<U>, U>(a: T, b: U) -> T::Result {
    a.vec_add(b)
}

/// Vector Subtract
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sub<T: sealed::VectorSub<U>, U>(a: T, b: U) -> T::Result {
    a.vec_sub(b)
}

/// Vector Multiply
///
/// ## Purpose
/// Compute the products of corresponding elements of two vectors.
///
/// ## Result value
/// Each element of r receives the product of the corresponding elements of a and b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mul<T: sealed::VectorMul>(a: T, b: T) -> T {
    a.vec_mul(b)
}

/// Vector Count Leading Zeros
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cntlz<T: sealed::CountBits>(a: T) -> T::Result {
    a.vec_cntlz()
}

/// Vector Count Trailing Zeros
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cnttz<T: sealed::CountBits>(a: T) -> T::Result {
    a.vec_cnttz()
}

/// Vector Population Count
///
/// Computes the population count (number of set bits) in each element of the input.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_popcnt<T: sealed::CountBits>(a: T) -> T::Result {
    a.vec_popcnt()
}

/// Vector Maximum
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_max<T: sealed::VectorMax<U>, U>(a: T, b: U) -> T::Result {
    a.vec_max(b)
}

/// Vector  Minimum
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_min<T: sealed::VectorMin<U>, U>(a: T, b: U) -> T::Result {
    a.vec_min(b)
}

/// Vector Absolute
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_abs<T: sealed::VectorAbs>(a: T) -> T {
    a.vec_abs()
}

/// Vector Negative Absolute
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nabs<T: sealed::VectorNabs>(a: T) -> T {
    a.vec_nabs()
}

/// Vector Negative Multiply Add
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nmadd<T: sealed::VectorNmadd>(a: T, b: T, c: T) -> T {
    a.vec_nmadd(b, c)
}

/// Vector Negative Multiply Subtract
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nmsub<T: sealed::VectorNmsub>(a: T, b: T, c: T) -> T {
    a.vec_nmsub(b, c)
}

/// Vector Square Root
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sqrt<T: sealed::VectorSqrt>(a: T) -> T {
    a.vec_sqrt()
}

/// Vector Splat
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_splat<T: sealed::VectorSplat, const IMM: u32>(a: T) -> T {
    a.vec_splat::<IMM>()
}

/// Vector Splats
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_splats<T: sealed::VectorSplats<U>, U>(a: T) -> U {
    a.vec_splats()
}

/// Vector AND
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_and<T: sealed::VectorAnd<U>, U>(a: T, b: U) -> T::Result {
    a.vec_and(b)
}

/// Vector OR
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_or<T: sealed::VectorOr<U>, U>(a: T, b: U) -> T::Result {
    a.vec_or(b)
}

/// Vector XOR
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_xor<T: sealed::VectorXor<U>, U>(a: T, b: U) -> T::Result {
    a.vec_xor(b)
}

/// Vector NOR
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nor<T: sealed::VectorNor<U>, U>(a: T, b: U) -> T::Result {
    a.vec_nor(b)
}

/// Vector NAND
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_nand<T: sealed::VectorNand<U>, U>(a: T, b: U) -> T::Result {
    a.vec_nand(b)
}

/// Vector XNOR
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_eqv<T: sealed::VectorEqv<U>, U>(a: T, b: U) -> T::Result {
    a.vec_eqv(b)
}

/// Vector ANDC
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_andc<T: sealed::VectorAndc<U>, U>(a: T, b: U) -> T::Result {
    a.vec_andc(b)
}

/// Vector OR with Complement
///
/// ## Purpose
/// Performs a bitwise OR of the first vector with the bitwise-complemented second vector.
///
/// ## Result value
/// r is the bitwise OR of a and the bitwise complement of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_orc<T: sealed::VectorOrc<U>, U>(a: T, b: U) -> T::Result {
    a.vec_orc(b)
}

/// Vector Floor
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_floor<T: sealed::VectorFloor>(a: T) -> T {
    a.vec_floor()
}

/// Vector Ceil
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_ceil<T: sealed::VectorCeil>(a: T) -> T {
    a.vec_ceil()
}

/// Vector Truncate
///
/// Returns a vector containing the truncated values of the corresponding elements of the given vector.
/// Each element of the result contains the value of the corresponding element of a, truncated to an integral value.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_trunc<T: sealed::VectorTrunc>(a: T) -> T {
    a.vec_trunc()
}

/// Vector Round
///
/// Returns a vector containing the rounded values to the nearest representable floating-point integer,
/// using IEEE round-to-nearest rounding, of the corresponding elements of the given vector
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_round<T: sealed::VectorRound>(a: T) -> T {
    a.vec_round()
}

/// Vector Round to Current
///
/// Returns a vector by using the current rounding mode to round every
/// floating-point element in the given vector to integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundc<T: sealed::VectorRoundc>(a: T) -> T {
    a.vec_roundc()
}

/// Vector Round toward Negative Infinity
///
/// Returns a vector containing the largest representable floating-point integral values less
/// than or equal to the values of the corresponding elements of the given vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundm<T: sealed::VectorFloor>(a: T) -> T {
    // the IBM docs note
    //
    // > vec_roundm provides the same functionality as vec_floor, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_floor` also does not trigger that exception, so both are equivalent
    a.vec_floor()
}

/// Vector Round toward Positive Infinity
///
/// Returns a vector containing the smallest representable floating-point integral values greater
/// than or equal to the values of the corresponding elements of the given vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundp<T: sealed::VectorCeil>(a: T) -> T {
    // the IBM docs note
    //
    // > vec_roundp provides the same functionality as vec_ceil, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_ceil` also does not trigger that exception, so both are equivalent
    a.vec_ceil()
}

/// Vector Round toward Zero
///
/// Returns a vector containing the truncated values of the corresponding elements of the given vector.
/// Each element of the result contains the value of the corresponding element of a, truncated to an integral value.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_roundz<T: sealed::VectorTrunc>(a: T) -> T {
    // the IBM docs note
    //
    // > vec_roundz provides the same functionality as vec_trunc, except that vec_roundz would not trigger the IEEE-inexact exception.
    //
    // but in practice `vec_trunc` also does not trigger that exception, so both are equivalent
    a.vec_trunc()
}

/// Vector Round to Integer
///
/// Returns a vector by using the current rounding mode to round every floating-point element in the given vector to integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rint<T: sealed::VectorRint>(a: T) -> T {
    a.vec_rint()
}

/// Vector Average
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_avg<T: sealed::VectorAvg<U>, U>(a: T, b: U) -> T::Result {
    a.vec_avg(b)
}

/// Vector Shift Left
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sl<T: sealed::VectorSl<U>, U>(a: T, b: U) -> T::Result {
    a.vec_sl(b)
}

/// Vector Shift Right
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sr<T: sealed::VectorSr<U>, U>(a: T, b: U) -> T::Result {
    a.vec_sr(b)
}

/// Vector Shift Right Algebraic
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sra<T: sealed::VectorSra<U>, U>(a: T, b: U) -> T::Result {
    a.vec_sra(b)
}

/// Vector Shift Left by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_slb<T: sealed::VectorSlb<U>, U>(a: T, b: U) -> T::Result {
    a.vec_slb(b)
}

/// Vector Shift Right by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srb<T: sealed::VectorSrb<U>, U>(a: T, b: U) -> T::Result {
    a.vec_srb(b)
}

/// Vector Shift Right Algebraic by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srab<T: sealed::VectorSrab<U>, U>(a: T, b: U) -> T::Result {
    a.vec_srab(b)
}

/// Vector Element Rotate Left
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rl<T: sealed::VectorRl<U>, U>(a: T, b: U) -> T::Result {
    a.vec_rl(b)
}

/// Vector Shift Left
///
/// Performs a left shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a left by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by zeros.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sll<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSll<vector_unsigned_char, Result = T>,
{
    a.vec_sll(b)
}

/// Vector Shift Right
///
/// Performs a right shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a right by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by zeros.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srl<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSrl<vector_unsigned_char, Result = T>,
{
    a.vec_srl(b)
}

/// Vector Shift Right Arithmetic
///
/// Performs an algebraic right shift for a vector by a given number of bits. Each element of the result is obtained by shifting the corresponding
/// element of a right by the number of bits specified by the last 3 bits of every byte of b. The bits that are shifted out are replaced by copies of
/// the most significant bit of the element of a.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sral<T>(a: T, b: vector_unsigned_char) -> T
where
    T: sealed::VectorSral<vector_unsigned_char, Result = T>,
{
    a.vec_sral(b)
}

/// Vector Element Rotate Left Immediate
///
/// Rotates each element of a vector left by a given number of bits. Each element of the result is obtained by rotating the corresponding element
/// of a left by the number of bits specified by b, modulo the number of bits in the element.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_rli<T: sealed::VectorRli>(a: T, bits: core::ffi::c_ulong) -> T {
    a.vec_rli(bits)
}

/// Vector Reverse Elements
///
/// Returns a vector with the elements of the input vector in reversed order.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_reve<T: sealed::VectorReve>(a: T) -> T {
    a.vec_reve()
}

/// Vector Byte Reverse
///
/// Returns a vector where each vector element contains the corresponding byte-reversed vector element of the input vector.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_revb<T: sealed::VectorRevb>(a: T) -> T {
    a.vec_revb()
}

/// Vector Merge High
///
/// Merges the most significant ("high") halves of two vectors.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mergeh<T: sealed::VectorMergeh>(a: T, b: T) -> T {
    a.vec_mergeh(b)
}

/// Vector Merge Low
///
/// Merges the least significant ("low") halves of two vectors.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mergel<T: sealed::VectorMergel>(a: T, b: T) -> T {
    a.vec_mergel(b)
}

/// Vector Pack
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_pack<T: sealed::VectorPack<U>, U>(a: T, b: U) -> T::Result {
    a.vec_pack(b)
}

/// Vector Pack Saturated
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_packs<T: sealed::VectorPacks<U>, U>(a: T, b: U) -> T::Result {
    a.vec_packs(b)
}

/// Vector Pack Saturated Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_packs_cc<T: sealed::VectorPacksCC>(a: T, b: T) -> (T::Result, i32) {
    a.vec_packs_cc(b)
}

/// Vector Pack Saturated Unsigned
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_packsu<T: sealed::VectorPacksu<U>, U>(a: T, b: U) -> T::Result {
    a.vec_packsu(b)
}

/// Vector Pack Saturated Unsigned Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_packsu_cc<T: sealed::VectorPacksuCC>(a: T, b: T) -> (T::Result, i32) {
    a.vec_packsu_cc(b)
}

/// Vector Unpack High
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_unpackh<T: sealed::VectorUnpackh>(a: T) -> <T as sealed::VectorUnpackh>::Result {
    a.vec_unpackh()
}

/// Vector Unpack Low
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_unpackl<T: sealed::VectorUnpackl>(a: T) -> <T as sealed::VectorUnpackl>::Result {
    a.vec_unpackl()
}

/// Vector Generate Byte Mask
///
/// Generates byte masks for elements in the return vector. For each bit in a, if the bit is one, all bit positions
/// in the corresponding byte element of d are set to ones. Otherwise, if the bit is zero, the corresponding byte element is set to zero.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgbm, MASK = 0x00FF))]
pub unsafe fn vec_genmask<const MASK: u16>() -> vector_unsigned_char {
    vector_unsigned_char(const { genmask::<MASK>() })
}

/// Vector Generate Mask (Byte)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, L = 3, H = 5))]
pub unsafe fn vec_genmasks_8<const L: u8, const H: u8>() -> vector_unsigned_char {
    vector_unsigned_char(const { [genmasks(u8::BITS, L, H) as u8; 16] })
}

/// Vector Generate Mask (Halfword)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, L = 3, H = 5))]
pub unsafe fn vec_genmasks_16<const L: u8, const H: u8>() -> vector_unsigned_short {
    vector_unsigned_short(const { [genmasks(u16::BITS, L, H) as u16; 8] })
}

/// Vector Generate Mask (Word)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgmf, L = 3, H = 5))]
pub unsafe fn vec_genmasks_32<const L: u8, const H: u8>() -> vector_unsigned_int {
    vector_unsigned_int(const { [genmasks(u32::BITS, L, H) as u32; 4] })
}

/// Vector Generate Mask (Doubleword)
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgmg, L = 3, H = 5))]
pub unsafe fn vec_genmasks_64<const L: u8, const H: u8>() -> vector_unsigned_long_long {
    vector_unsigned_long_long(const { [genmasks(u64::BITS, L, H); 2] })
}

/// Vector Permute
///
/// Returns a vector that contains some elements of two vectors, in the order specified by a third vector.
/// Each byte of the result is selected by using the least significant 5 bits of the corresponding byte of c as an index into the concatenated bytes of a and b.
/// Note: The vector generate mask built-in function [`vec_genmask`] could help generate the mask c.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_perm<T: sealed::VectorPerm>(a: T, b: T, c: vector_unsigned_char) -> T {
    a.vec_perm(b, c)
}

/// Vector Sum Across Quadword
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the quadword of vector a,
/// and the rightmost word or doubleword element of the b. The result is an unsigned 128-bit integer.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum_u128<T: sealed::VectorSumU128>(a: T, b: T) -> vector_unsigned_char {
    a.vec_sum_u128(b)
}

/// Vector Sum Across Doubleword
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the doubleword of vector a,
/// and the rightmost sub-element of the corresponding doubleword of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum2<T: sealed::VectorSum2>(a: T, b: T) -> vector_unsigned_long_long {
    a.vec_sum2(b)
}

/// Vector Sum Across Word
///
/// Returns a vector containing the results of performing a sum across all the elements in each of the word of vector a,
/// and the rightmost sub-element of the corresponding word of b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sum4<T: sealed::VectorSum4>(a: T, b: T) -> vector_unsigned_int {
    a.vec_sum4(b)
}

/// Vector Addition unsigned 128-bits
///
/// Adds unsigned quadword values.
///
/// This function operates on the vectors as 128-bit unsigned integers. It returns low 128 bits of a + b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vaq))]
pub unsafe fn vec_add_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);
    transmute(a.wrapping_add(b))
}

/// Vector Subtract unsigned 128-bits
///
/// Subtracts unsigned quadword values.
///
/// This function operates on the vectors as 128-bit unsigned integers. It returns low 128 bits of a - b.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsq))]
pub unsafe fn vec_sub_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);

    transmute(a.wrapping_sub(b))
}

/// Vector Subtract Carryout
///
/// Returns a vector containing the borrow produced by subtracting each of corresponding elements of b from a.
///
/// On each resulting element, the value is 0 if a borrow occurred, or 1 if no borrow occurred.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_subc<T: sealed::VectorSubc<U>, U>(a: T, b: U) -> T::Result {
    a.vec_subc(b)
}

/// Vector Subtract Carryout unsigned 128-bits
///
/// Gets the carry bit of the 128-bit subtraction of two quadword values.
/// This function operates on the vectors as 128-bit unsigned integers. It returns a vector containing the borrow produced by subtracting b from a, as unsigned 128-bits integers.
/// If no borrow occurred, the bit 127 of d is 1; otherwise it is set to 0. All other bits of d are 0.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vscbiq))]
pub unsafe fn vec_subc_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);
    transmute(!a.overflowing_sub(b).1 as u128)
}

/// Vector Add Compute Carryout unsigned 128-bits
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vaccq))]
pub unsafe fn vec_addc_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);
    // FIXME(llvm) https://github.com/llvm/llvm-project/pull/153557
    // transmute(a.overflowing_add(b).1 as u128)
    transmute(vaccq(a, b))
}

/// Vector Add With Carry unsigned 128-bits
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vacq))]
pub unsafe fn vec_adde_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);
    let c: u128 = transmute(c);
    // FIXME(llvm) https://github.com/llvm/llvm-project/pull/153557
    //     let (d, _carry) = a.carrying_add(b, c & 1 != 0);
    //     transmute(d)
    transmute(vacq(a, b, c))
}

/// Vector Add With Carry Compute Carry unsigned 128-bits
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vacccq))]
pub unsafe fn vec_addec_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    let a: u128 = transmute(a);
    let b: u128 = transmute(b);
    let c: u128 = transmute(c);
    // FIXME(llvm) https://github.com/llvm/llvm-project/pull/153557
    // let (_d, carry) = a.carrying_add(b, c & 1 != 0);
    // transmute(carry as u128)
    transmute(vacccq(a, b, c))
}

/// Vector Subtract with Carryout
///
/// Subtracts unsigned quadword values with carry bit from a previous operation.
///
/// This function operates on the vectors as 128-bit unsigned integers. It returns a vector containing the result of subtracting of b from a,
/// and the carryout bit from a previous operation.
///
/// Note: Only the borrow indication bit (127-bit) of c is used, and the other bits are ignored.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsbiq))]
pub unsafe fn vec_sube_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vsbiq(transmute(a), transmute(b), transmute(c)))
}

/// Vector Subtract with Carryout, Carryout
///
/// Gets the carry bit of the 128-bit subtraction of two quadword values with carry bit from the previous operation.
///
/// It returns a vector containing the carryout produced from the result of subtracting of b from a,
/// and the carryout bit from a previous operation. If no borrow occurred, the 127-bit of d is 1, otherwise 0.
/// All other bits of d are 0.
///
/// Note: Only the borrow indication bit (127-bit) of c is used, and the other bits are ignored.
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vsbcbiq))]
pub unsafe fn vec_subec_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vsbcbiq(transmute(a), transmute(b), transmute(c)))
}

/// Vector Splat Signed Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, IMM = 42))]
pub unsafe fn vec_splat_s8<const IMM: i8>() -> vector_signed_char {
    vector_signed_char([IMM; 16])
}

/// Vector Splat Signed Halfword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, IMM = 42))]
pub unsafe fn vec_splat_s16<const IMM: i16>() -> vector_signed_short {
    vector_signed_short([IMM; 8])
}

/// Vector Splat Signed Word
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepif, IMM = 42))]
pub unsafe fn vec_splat_s32<const IMM: i16>() -> vector_signed_int {
    vector_signed_int([IMM as i32; 4])
}

/// Vector Splat Signed Doubleword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepig, IMM = 42))]
pub unsafe fn vec_splat_s64<const IMM: i16>() -> vector_signed_long_long {
    vector_signed_long_long([IMM as i64; 2])
}

/// Vector Splat Unsigned Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepib, IMM = 42))]
pub unsafe fn vec_splat_u8<const IMM: u8>() -> vector_unsigned_char {
    vector_unsigned_char([IMM; 16])
}

/// Vector Splat Unsigned Halfword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepih, IMM = 42))]
pub unsafe fn vec_splat_u16<const IMM: i16>() -> vector_unsigned_short {
    vector_unsigned_short([IMM as u16; 8])
}

/// Vector Splat Unsigned Word
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepif, IMM = 42))]
pub unsafe fn vec_splat_u32<const IMM: i16>() -> vector_unsigned_int {
    vector_unsigned_int([IMM as u32; 4])
}

/// Vector Splat Unsigned Doubleword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vrepig, IMM = 42))]
pub unsafe fn vec_splat_u64<const IMM: i16>() -> vector_unsigned_long_long {
    vector_unsigned_long_long([IMM as u64; 2])
}

macro_rules! vec_find_any {
    ($($Trait:ident $fun:ident $doc:literal)*) => {
        $(
            #[inline]
            #[target_feature(enable = "vector")]
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            #[doc = $doc]
            pub unsafe fn $fun<T: sealed::$Trait<U>, U>(a: T, b: U) -> T::Result {
                a.$fun(b)
            }
        )*
    }
}

vec_find_any! {
    VectorFindAnyEq vec_find_any_eq "Vector Find Any Element Equal with Condition Code"
    VectorFindAnyNe vec_find_any_ne "Vector Find Any Element Not Equal with Condition Code"
    VectorFindAnyEqIdx vec_find_any_eq_idx "Vector Find Any Element Equal Index with Condition Code"
    VectorFindAnyNeIdx vec_find_any_ne_idx "Vector Find Any Element Not Equal Index with Condition Code"
    VectorFindAnyEqOrZeroIdx vec_find_any_eq_or_0_idx "Vector Find Any Element Equal or Zero Index with Condition Code"
    VectorFindAnyNeOrZeroIdx vec_find_any_ne_or_0_idx "Vector Find Any Element Not Equal or Zero Index with Condition Code"
}

macro_rules! vec_find_any_cc {
    ($($Trait:ident $fun:ident $doc:literal)*) => {
        $(
            #[inline]
            #[target_feature(enable = "vector")]
            #[unstable(feature = "stdarch_s390x", issue = "135681")]
            #[doc = $doc]
            pub unsafe fn $fun<T: sealed::$Trait<U>, U>(a: T, b: U) -> (T::Result, i32) {
                a.$fun(b)
            }
        )*
    }
}

vec_find_any_cc! {
    VectorFindAnyEqCC vec_find_any_eq_cc "Vector Find Any Element Equal with Condition Code"
    VectorFindAnyNeCC vec_find_any_ne_cc "Vector Find Any Element Not Equal with Condition Code"
    VectorFindAnyEqIdxCC vec_find_any_eq_idx_cc "Vector Find Any Element Equal Index with Condition Code"
    VectorFindAnyNeIdxCC vec_find_any_ne_idx_cc "Vector Find Any Element Not Equal Index with Condition Code"
    VectorFindAnyEqOrZeroIdxCC vec_find_any_eq_or_0_idx_cc "Vector Find Any Element Equal or Zero Index with Condition Code"
    VectorFindAnyNeOrZeroIdxCC vec_find_any_ne_or_0_idx_cc "Vector Find Any Element Not Equal or Zero Index with Condition Code"
}

/// Vector Load
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_xl<T: sealed::VectorLoad>(offset: isize, ptr: *const T::ElementType) -> T {
    T::vec_xl(offset, ptr)
}

/// Vector Load Pair
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_load_pair<T: sealed::VectorLoadPair>(a: T::ElementType, b: T::ElementType) -> T {
    T::vec_load_pair(a, b)
}

/// Vector Load to Block Boundary
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_load_bndry<T: sealed::VectorLoad, const BLOCK_BOUNDARY: u16>(
    ptr: *const T::ElementType,
) -> MaybeUninit<T> {
    T::vec_load_bndry::<BLOCK_BOUNDARY>(ptr)
}

/// Vector Store
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_xst<T: sealed::VectorStore>(vector: T, offset: isize, ptr: *mut T::ElementType) {
    vector.vec_xst(offset, ptr)
}

/// Vector Load with Length
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_load_len<T: sealed::VectorLoad>(
    ptr: *const T::ElementType,
    byte_count: u32,
) -> T {
    T::vec_load_len(ptr, byte_count)
}

/// Vector Store with Length
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_store_len<T: sealed::VectorStore>(
    vector: T,
    ptr: *mut T::ElementType,
    byte_count: u32,
) {
    vector.vec_store_len(ptr, byte_count)
}

/// Vector Load Rightmost with Length
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vlrlr))]
pub unsafe fn vec_load_len_r(ptr: *const u8, byte_count: u32) -> vector_unsigned_char {
    vlrl(byte_count, ptr)
}

/// Vector Store Rightmost with Length
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vstrlr))]
pub unsafe fn vec_store_len_r(vector: vector_unsigned_char, ptr: *mut u8, byte_count: u32) {
    vstrl(vector, byte_count, ptr)
}

/// Vector Multiply Add
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_madd<T: sealed::VectorMadd>(a: T, b: T, c: T) -> T {
    a.vec_madd(b, c)
}

/// Vector Multiply Add
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_msub<T: sealed::VectorMadd>(a: T, b: T, c: T) -> T {
    a.vec_msub(b, c)
}

/// Vector Multiply and Add Even
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_meadd<T: sealed::VectorMeadd>(a: T, b: T, c: T::Result) -> T::Result {
    a.vec_meadd(b, c)
}

/// Vector Multiply and Add Odd
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_moadd<T: sealed::VectorMoadd>(a: T, b: T, c: T::Result) -> T::Result {
    a.vec_moadd(b, c)
}

/// Vector Multiply and Add High
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mhadd<T: sealed::VectorMhadd>(a: T, b: T, c: T::Result) -> T::Result {
    a.vec_mhadd(b, c)
}

/// Vector Multiply and Add Low
#[inline]
#[target_feature(enable = "vector-packed-decimal")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mladd<T: sealed::VectorMladd>(a: T, b: T, c: T::Result) -> T::Result {
    a.vec_mladd(b, c)
}

/// Vector Checksum
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vcksm))]
pub unsafe fn vec_checksum(a: vector_unsigned_int, b: vector_unsigned_int) -> vector_unsigned_int {
    vcksm(a, b)
}

/// Vector Multiply Even
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mule<T: sealed::VectorMule<U>, U>(a: T, b: T) -> U {
    a.vec_mule(b)
}

/// Vector Multiply Odd
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mulo<T: sealed::VectorMulo<U>, U>(a: T, b: T) -> U {
    a.vec_mulo(b)
}

/// Vector Multiply High
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_mulh<T: sealed::VectorMulh<U>, U>(a: T, b: T) -> U {
    a.vec_mulh(b)
}

/// Vector Galois Field Multiply Sum
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_gfmsum<T: sealed::VectorGfmsum<U>, U>(a: T, b: T) -> U {
    a.vec_gfmsum(b)
}

/// Vector Galois Field Multiply Sum
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_gfmsum_accum<T: sealed::VectorGfmsumAccum>(
    a: T,
    b: T,
    c: T::Result,
) -> T::Result {
    a.vec_gfmsum_accum(b, c)
}

/// Vector Galois Field Multiply Sum 128-bits
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgfmg))]
pub unsafe fn vec_gfmsum_128(
    a: vector_unsigned_long_long,
    b: vector_unsigned_long_long,
) -> vector_unsigned_char {
    transmute(vgfmg(a, b))
}

/// Vector Galois Field Multiply Sum and Accumulate 128-bits
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vgfmag))]
pub unsafe fn vec_gfmsum_accum_128(
    a: vector_unsigned_long_long,
    b: vector_unsigned_long_long,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    transmute(vgfmag(a, b, transmute(c)))
}

/// Vector Bit Permute
#[inline]
#[target_feature(enable = "vector-enhancements-1")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(test, assert_instr(vbperm))]
pub unsafe fn vec_bperm_u128(
    a: vector_unsigned_char,
    b: vector_unsigned_char,
) -> vector_unsigned_long_long {
    vbperm(a, b)
}

/// Vector Gather Element
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_gather_element<T: sealed::VectorGatherElement, const D: u32>(
    a: T,
    b: T::Offset,
    c: *const T::Element,
) -> T {
    a.vec_gather_element::<D>(b, c)
}

/// Vector Select
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sel<T: sealed::VectorSel<U>, U>(a: T, b: T, c: U) -> T {
    a.vec_sel(b, c)
}

#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_ZERO_P: u32 = 1 << 11;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_ZERO_N: u32 = 1 << 10;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_ZERO: u32 = __VEC_CLASS_FP_ZERO_P | __VEC_CLASS_FP_ZERO_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_NORMAL_P: u32 = 1 << 9;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_NORMAL_N: u32 = 1 << 8;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_NORMAL: u32 = __VEC_CLASS_FP_NORMAL_P | __VEC_CLASS_FP_NORMAL_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SUBNORMAL_P: u32 = 1 << 7;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SUBNORMAL_N: u32 = 1 << 6;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SUBNORMAL: u32 = __VEC_CLASS_FP_SUBNORMAL_P | __VEC_CLASS_FP_SUBNORMAL_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_INFINITY_P: u32 = 1 << 5;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_INFINITY_N: u32 = 1 << 4;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_INFINITY: u32 = __VEC_CLASS_FP_INFINITY_P | __VEC_CLASS_FP_INFINITY_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_QNAN_P: u32 = 1 << 3;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_QNAN_N: u32 = 1 << 2;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_QNAN: u32 = __VEC_CLASS_FP_QNAN_P | __VEC_CLASS_FP_QNAN_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SNAN_P: u32 = 1 << 1;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SNAN_N: u32 = 1 << 0;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_SNAN: u32 = __VEC_CLASS_FP_SNAN_P | __VEC_CLASS_FP_SNAN_N;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_NAN: u32 = __VEC_CLASS_FP_QNAN | __VEC_CLASS_FP_SNAN;
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub const __VEC_CLASS_FP_NOT_NORMAL: u32 =
    __VEC_CLASS_FP_NAN | __VEC_CLASS_FP_SUBNORMAL | __VEC_CLASS_FP_ZERO | __VEC_CLASS_FP_INFINITY;

/// Vector Floating-Point Test Data Class
///
/// You can use the `__VEC_CLASS_FP_*` constants as the argument for this operand
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_fp_test_data_class<T: sealed::VectorFpTestDataClass, const CLASS: u32>(
    a: T,
    c: *mut i32,
) -> T::Result {
    let (x, y) = a.vec_fp_test_data_class::<CLASS>();
    c.write(y);
    x
}

/// All Elements Not a Number
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_nan<T: sealed::VectorFpTestDataClass>(a: T) -> i32 {
    i32::from(a.vec_fp_test_data_class::<__VEC_CLASS_FP_NAN>().1 == 0)
}

/// All Elements Numeric
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_numeric<T: sealed::VectorFpTestDataClass>(a: T) -> i32 {
    i32::from(a.vec_fp_test_data_class::<__VEC_CLASS_FP_NAN>().1 == 3)
}

/// Any Elements Not a Number
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_nan<T: sealed::VectorFpTestDataClass>(a: T) -> i32 {
    i32::from(a.vec_fp_test_data_class::<__VEC_CLASS_FP_NAN>().1 != 3)
}

/// Any Elements Numeric
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_numeric<T: sealed::VectorFpTestDataClass>(a: T) -> i32 {
    i32::from(a.vec_fp_test_data_class::<__VEC_CLASS_FP_NAN>().1 != 0)
}

/// Vector Test under Mask
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_test_mask<T: sealed::VectorTestMask>(a: T, b: T::Mask) -> i32 {
    // I can't find much information about this, but this might just be a check for whether the
    // bitwise and of a and b is non-zero?
    a.vec_test_mask(b)
}

/// Vector Search String
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_search_string_cc<T: sealed::VectorSearchString>(
    a: T,
    b: T,
    c: vector_unsigned_char,
) -> (vector_unsigned_char, i32) {
    a.vec_search_string_cc(b, c)
}

/// Vector Search String Until Zero
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_search_string_until_zero_cc<T: sealed::VectorSearchString>(
    a: T,
    b: T,
    c: vector_unsigned_char,
) -> (vector_unsigned_char, i32) {
    a.vec_search_string_until_zero_cc(b, c)
}

/// Vector Convert from float (even elements) to double
#[inline]
#[target_feature(enable = "vector-enhancements-1")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
// FIXME: this emits `vflls` where `vldeb` is expected
// #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vldeb))]
pub unsafe fn vec_doublee(a: vector_float) -> vector_double {
    let even = simd_shuffle::<_, _, f32x2>(a, a, const { u32x2::from_array([0, 2]) });
    simd_as(even)
}

/// Vector Convert from double to float (even elements)
#[inline]
#[target_feature(enable = "vector-enhancements-1")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
// FIXME: the C version uses a shuffle mask with poison; we can't do that
// #[cfg_attr(all(test, target_feature = "vector-enhancements-1"), assert_instr(vledb))]
pub unsafe fn vec_floate(a: vector_double) -> vector_float {
    let truncated: f32x2 = simd_as(a);
    simd_shuffle(
        truncated,
        truncated,
        const { u32x4::from_array([0, 0, 1, 1]) },
    )
}

/// Vector Convert from int to float
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_float(a: impl sealed::VectorFloat) -> vector_float {
    a.vec_float()
}

/// Vector Convert from long long to double
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_double(a: impl sealed::VectorDouble) -> vector_double {
    a.vec_double()
}

/// Vector Sign Extend to Doubleword
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_extend_s64(a: impl sealed::VectorExtendSigned64) -> vector_signed_long_long {
    a.vec_extend_s64()
}

/// Vector Convert floating point to signed
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_signed<T: sealed::VectorSigned>(a: T) -> T::Result {
    a.vec_signed()
}

/// Vector Convert floating point to unsigned
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_unsigned<T: sealed::VectorUnsigned>(a: T) -> T::Result {
    a.vec_unsigned()
}

/// Vector Copy Until Zero
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cp_until_zero<T: sealed::VectorCopyUntilZero>(a: T) -> T {
    a.vec_cp_until_zero()
}

/// Vector Copy Until Zero
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cp_until_zero_cc<T: sealed::VectorCopyUntilZeroCC>(a: T) -> (T, i32) {
    a.vec_cp_until_zero_cc()
}

/// Vector Multiply Sum Logical
#[inline]
#[target_feature(enable = "vector-enhancements-1")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
#[cfg_attr(
    all(test, target_feature = "vector-enhancements-1"),
    assert_instr(vmslg, D = 4)
)]
pub unsafe fn vec_msum_u128<const D: u32>(
    a: vector_unsigned_long_long,
    b: vector_unsigned_long_long,
    c: vector_unsigned_char,
) -> vector_unsigned_char {
    const {
        if !matches!(D, 0 | 4 | 8 | 12) {
            panic!("D needs to be one of 0, 4, 8, 12");
        }
    };
    transmute(vmslg(a, b, transmute(c), D))
}

/// Vector Shift Left Double by Byte
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sld<T: sealed::VectorSld, const C: u32>(a: T, b: T) -> T {
    static_assert_uimm_bits!(C, 4);
    a.vec_sld::<C>(b)
}

/// Vector Shift Left Double by Word
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sldw<T: sealed::VectorSld, const C: u32>(a: T, b: T) -> T {
    static_assert_uimm_bits!(C, 2);
    a.vec_sldw::<C>(b)
}

/// Vector Shift Left Double by Bit
#[inline]
#[target_feature(enable = "vector-enhancements-2")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_sldb<T: sealed::VectorSld, const C: u32>(a: T, b: T) -> T {
    static_assert_uimm_bits!(C, 3);
    a.vec_sldb::<C>(b)
}

/// Vector Shift Right Double by Bit
#[inline]
#[target_feature(enable = "vector-enhancements-2")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_srdb<T: sealed::VectorSrdb, const C: u32>(a: T, b: T) -> T {
    static_assert_uimm_bits!(C, 3);
    a.vec_srdb::<C>(b)
}

/// Vector Compare Ranges
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrc::<{ FindImm::Eq as u32 }>(b, c)
}

/// Vector Compare Not in Ranges
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrc::<{ FindImm::Ne as u32 }>(b, c)
}

/// Vector Compare Ranges Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg_idx<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrc::<{ FindImm::EqIdx as u32 }>(b, c)
}

/// Vector Compare Not in Ranges Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg_idx<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrc::<{ FindImm::NeIdx as u32 }>(b, c)
}

/// Vector Compare Ranges with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrcs::<{ FindImm::Eq as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Not in Ranges with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrcs::<{ FindImm::Ne as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Ranges Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg_idx_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrcs::<{ FindImm::EqIdx as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Not in Ranges Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg_idx_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrcs::<{ FindImm::NeIdx as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Ranges or Zero Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg_or_0_idx<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrcz::<{ FindImm::EqIdx as u32 }>(b, c)
}

/// Vector Compare Not in Ranges or Zero Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg_or_0_idx<T: sealed::VectorCompareRange>(a: T, b: T, c: T) -> T::Result {
    a.vstrcz::<{ FindImm::NeIdx as u32 }>(b, c)
}

/// Vector Compare Ranges or Zero Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmprg_or_0_idx_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrczs::<{ FindImm::EqIdx as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Not in Ranges or Zero Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpnrg_or_0_idx_cc<T: sealed::VectorCompareRange>(
    a: T,
    b: T,
    c: T,
    d: *mut i32,
) -> T::Result {
    let (x, y) = a.vstrczs::<{ FindImm::NeIdx as u32 }>(b, c);
    d.write(y);
    x
}

/// Vector Compare Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpeq<T: sealed::VectorEquality>(a: T, b: T) -> T::Result {
    a.vec_cmpeq(b)
}

/// Vector Compare Not Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpne<T: sealed::VectorEquality>(a: T, b: T) -> T::Result {
    a.vec_cmpne(b)
}

/// Vector Compare Greater Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpgt<T: sealed::VectorComparePredicate>(a: T, b: T) -> T::Result {
    a.vec_cmpgt(b)
}

/// Vector Compare Greater Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpge<T: sealed::VectorComparePredicate>(a: T, b: T) -> T::Result {
    a.vec_cmpge(b)
}

/// Vector Compare Less
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmplt<T: sealed::VectorComparePredicate>(a: T, b: T) -> T::Result {
    a.vec_cmplt(b)
}

/// Vector Compare Less Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmple<T: sealed::VectorComparePredicate>(a: T, b: T) -> T::Result {
    a.vec_cmple(b)
}

/// Vector Compare Equal Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpeq_idx<T: sealed::VectorEqualityIdx>(a: T, b: T) -> T::Result {
    a.vec_cmpeq_idx(b)
}
/// Vector Compare Not Equal Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpne_idx<T: sealed::VectorEqualityIdx>(a: T, b: T) -> T::Result {
    a.vec_cmpne_idx(b)
}
/// Vector Compare Equal Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpeq_idx_cc<T: sealed::VectorEqualityIdx>(a: T, b: T) -> (T::Result, i32) {
    a.vec_cmpeq_idx_cc(b)
}
/// Vector Compare Not Equal Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpne_idx_cc<T: sealed::VectorEqualityIdx>(a: T, b: T) -> (T::Result, i32) {
    a.vec_cmpne_idx_cc(b)
}
/// Vector Compare Equal or Zero Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpeq_or_0_idx<T: sealed::VectorEqualityIdx>(a: T, b: T) -> T::Result {
    a.vec_cmpeq_or_0_idx(b)
}
/// Vector Compare Not Equal or Zero Index
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpne_or_0_idx<T: sealed::VectorEqualityIdx>(a: T, b: T) -> T::Result {
    a.vec_cmpne_or_0_idx(b)
}
/// Vector Compare Equal or Zero Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpeq_or_0_idx_cc<T: sealed::VectorEqualityIdx>(a: T, b: T) -> (T::Result, i32) {
    a.vec_cmpeq_or_0_idx_cc(b)
}
/// Vector Compare Not Equal or Zero Index with Condition Code
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_cmpne_or_0_idx_cc<T: sealed::VectorEqualityIdx>(a: T, b: T) -> (T::Result, i32) {
    a.vec_cmpne_or_0_idx_cc(b)
}

/// All Elements Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_eq<T: sealed::VectorEquality>(a: T, b: T) -> i32 {
    simd_reduce_all(vec_cmpeq(a, b)) as i32 as i32
}

/// All Elements Not Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_ne<T: sealed::VectorEquality>(a: T, b: T) -> i32 {
    simd_reduce_all(vec_cmpne(a, b)) as i32
}

/// Any Element Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_eq<T: sealed::VectorEquality>(a: T, b: T) -> i32 {
    simd_reduce_any(vec_cmpeq(a, b)) as i32
}

/// Any Element Not Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_ne<T: sealed::VectorEquality>(a: T, b: T) -> i32 {
    simd_reduce_any(vec_cmpne(a, b)) as i32
}

/// All Elements Less Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_lt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    a.vec_all_lt(b)
}

/// All Elements Less Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_le<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    a.vec_all_le(b)
}

/// All Elements Greater Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_gt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    a.vec_all_gt(b)
}

/// All Elements Greater Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_ge<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    a.vec_all_ge(b)
}

/// All Elements Not Less Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_nlt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_all_ge(a, b)
}

/// All Elements Not Less Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_nle<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_all_gt(a, b)
}

/// All Elements Not Greater Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_ngt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_all_le(a, b)
}

/// All Elements Not Greater Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_all_nge<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_all_lt(a, b)
}

/// Any Elements Less Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_lt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    !vec_all_ge(a, b)
}

/// Any Elements Less Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_le<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    !vec_all_gt(a, b)
}

/// Any Elements Greater Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_gt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    !vec_all_le(a, b)
}

/// Any Elements Greater Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_ge<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    !vec_all_lt(a, b)
}

/// Any Elements Not Less Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_nlt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_any_ge(a, b)
}

/// Any Elements Not Less Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_nle<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_any_gt(a, b)
}

/// Any Elements Not Greater Than
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_ngt<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_any_le(a, b)
}

/// Any Elements Not Greater Than or Equal
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_any_nge<T: sealed::VectorCompare>(a: T, b: T) -> i32 {
    vec_any_lt(a, b)
}

/// Vector Extract
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_extract<T: sealed::VectorExtract>(a: T, b: i32) -> T::ElementType {
    T::vec_extract(a, b)
}

/// Vector Insert
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_insert<T: sealed::VectorInsert>(a: T::ElementType, b: T, c: i32) -> T {
    T::vec_insert(a, b, c)
}

/// Vector Insert and Zero
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_insert_and_zero<T: sealed::VectorInsertAndZero>(a: *const T::ElementType) -> T {
    T::vec_insert_and_zero(a)
}

/// Vector Promote
#[inline]
#[target_feature(enable = "vector")]
#[unstable(feature = "stdarch_s390x", issue = "135681")]
pub unsafe fn vec_promote<T: sealed::VectorPromote>(a: T::ElementType, b: i32) -> MaybeUninit<T> {
    T::vec_promote(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    impl<const N: usize> ShuffleMask<N> {
        fn as_array(&self) -> &[u32; N] {
            unsafe { std::mem::transmute(self) }
        }
    }

    #[test]
    fn reverse_mask() {
        assert_eq!(ShuffleMask::<4>::reverse().as_array(), &[3, 2, 1, 0]);
    }

    #[test]
    fn mergel_mask() {
        assert_eq!(ShuffleMask::<4>::merge_low().as_array(), &[2, 6, 3, 7]);
    }

    #[test]
    fn mergeh_mask() {
        assert_eq!(ShuffleMask::<4>::merge_high().as_array(), &[0, 4, 1, 5]);
    }

    #[test]
    fn pack_mask() {
        assert_eq!(ShuffleMask::<4>::pack().as_array(), &[1, 3, 5, 7]);
    }

    #[test]
    fn test_vec_mask() {
        assert_eq!(
            genmask::<0x00FF>(),
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
            ]
        );
    }

    #[test]
    fn test_genmasks() {
        assert_eq!(genmasks(u8::BITS, 3, 5), 28);
        assert_eq!(genmasks(u8::BITS, 3, 7), 31);

        // If a or b is greater than 8, the operation is performed as if the value gets modulo by 8.
        assert_eq!(genmasks(u8::BITS, 3 + 8, 7 + 8), 31);
        // If a is greater than b, the operation is perform as if b equals 7.
        assert_eq!(genmasks(u8::BITS, 5, 4), genmasks(u8::BITS, 5, 7));

        assert_eq!(
            genmasks(u16::BITS, 4, 12) as u16,
            u16::from_be_bytes([15, -8i8 as u8])
        );
        assert_eq!(
            genmasks(u32::BITS, 4, 29) as u32,
            u32::from_be_bytes([15, 0xFF, 0xFF, -4i8 as u8])
        );
    }

    macro_rules! test_vec_1 {
        { $name: ident, $fn:ident, f32x4, [$($a:expr),+], ~[$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: vector_float = transmute(f32x4::new($($a),+));

                let d: vector_float = transmute(f32x4::new($($d),+));
                let r = transmute(vec_cmple(vec_abs(vec_sub($fn(a), d)), vec_splats(f32::EPSILON)));
                let e = m32x4::new(true, true, true, true);
                assert_eq!(e, r);
            }
        };
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($d:expr),+] } => {
            test_vec_1! { $name, $fn, $ty -> $ty, [$($a),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a));
                assert_eq!(d, r);
            }
        }
    }

    macro_rules! test_vec_2 {
        { $name: ident, $fn:ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! { $name, $fn, $ty -> $ty, [$($a),+], [$($b),+], [$($d),+] }
        };
        { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! { $name, $fn, $ty, $ty -> $ty, [$($a),+], [$($b),+], [$($d),+] }
         };
        { $name: ident, $fn:ident, $ty1: ident, $ty2: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty1) = transmute($ty1::new($($a),+));
                let b: s_t_l!($ty2) = transmute($ty2::new($($b),+));

                let d = $ty_out::new($($d),+);
                let r : $ty_out = transmute($fn(a, b));
                assert_eq!(d, r);
            }
         };
         { $name: ident, $fn:ident, $ty: ident -> $ty_out: ident, [$($a:expr),+], [$($b:expr),+], $d:expr } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = transmute($ty::new($($a),+));
                let b: s_t_l!($ty) = transmute($ty::new($($b),+));

                let r : $ty_out = transmute($fn(a, b));
                assert_eq!($d, r);
            }
         }
   }

    #[simd_test(enable = "vector")]
    unsafe fn vec_add_i32x4_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        let y = i32x4::new(4, 3, 2, 1);
        let x: vector_signed_int = transmute(x);
        let y: vector_signed_int = transmute(y);
        let z = vec_add(x, y);
        assert_eq!(i32x4::splat(5), transmute(z));
    }

    macro_rules! test_vec_sub {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_sub, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_sub! { test_vec_sub_f32x4, f32x4,
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, -1.0, -2.0],
    [-3.0, -1.0, 2.0, 4.0] }

    test_vec_sub! { test_vec_sub_f64x2, f64x2,
    [-1.0, 0.0],
    [2.0, 1.0],
    [-3.0, -1.0] }

    test_vec_sub! { test_vec_sub_i64x2, i64x2,
    [-1, 0],
    [2, 1],
    [-3, -1] }

    test_vec_sub! { test_vec_sub_u64x2, u64x2,
    [0, 1],
    [1, 0],
    [u64::MAX, 1] }

    test_vec_sub! { test_vec_sub_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u32x4, u32x4,
    [0, 0, 1, 2],
    [2, 1, 0, 0],
    [4294967294, 4294967295, 1, 2] }

    test_vec_sub! { test_vec_sub_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u16x8, u16x8,
    [0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0],
    [65534, 65535, 1, 2, 65534, 65535, 1, 2] }

    test_vec_sub! { test_vec_sub_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4, -3, -1, 2, 4] }

    test_vec_sub! { test_vec_sub_u8x16, u8x16,
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0],
    [254, 255, 1, 2, 254, 255, 1, 2, 254, 255, 1, 2, 254, 255, 1, 2] }

    macro_rules! test_vec_mul {
        { $name: ident, $ty: ident, [$($a:expr),+], [$($b:expr),+], [$($d:expr),+] } => {
            test_vec_2! {$name, vec_mul, $ty, [$($a),+], [$($b),+], [$($d),+] }
        }
    }

    test_vec_mul! { test_vec_mul_f32x4, f32x4,
    [-1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, -1.0, -2.0],
    [-2.0, 0.0, -1.0, -4.0] }

    test_vec_mul! { test_vec_mul_f64x2, f64x2,
    [-1.0, 0.0],
    [2.0, 1.0],
    [-2.0, 0.0] }

    test_vec_mul! { test_vec_mul_i64x2, i64x2,
    [i64::MAX, -4],
    [2, 3],
    [i64::MAX.wrapping_mul(2), -12] }

    test_vec_mul! { test_vec_mul_u64x2, u64x2,
    [u64::MAX, 4],
    [2, 3],
    [u64::MAX.wrapping_mul(2), 12] }

    test_vec_mul! { test_vec_mul_i32x4, i32x4,
    [-1, 0, 1, 2],
    [2, 1, -1, -2],
    [-2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u32x4, u32x4,
    [0, u32::MAX - 1, 1, 2],
    [5, 6, 7, 8],
    [0, 4294967284, 7, 16] }

    test_vec_mul! { test_vec_mul_i16x8, i16x8,
    [-1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2],
    [-2, 0, -1, -4, -2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u16x8, u16x8,
    [0, u16::MAX - 1, 1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 9, 8, 7, 6],
    [0, 65524, 7, 16, 27, 32, 35, 36] }

    test_vec_mul! { test_vec_mul_i8x16, i8x16,
    [-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2],
    [2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2],
    [-2, 0, -1, -4, -2, 0, -1, -4, -2, 0, -1, -4, -2, 0, -1, -4] }

    test_vec_mul! { test_vec_mul_u8x16, u8x16,
    [0, u8::MAX - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4],
    [5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 0, u8::MAX, 1, 2, 3, 4],
    [0, 244, 7, 16, 27, 32, 35, 36, 35, 32, 0, 248, 7, 12, 15, 16] }

    macro_rules! test_vec_abs {
        { $name: ident, $ty: ident, $a: expr, $d: expr } => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: s_t_l!($ty) = vec_splats($a);
                let a: s_t_l!($ty) = vec_abs(a);
                let d = $ty::splat($d);
                assert_eq!(d, transmute(a));
            }
        }
    }

    test_vec_abs! { test_vec_abs_i8, i8x16, -42i8, 42i8 }
    test_vec_abs! { test_vec_abs_i16, i16x8, -42i16, 42i16 }
    test_vec_abs! { test_vec_abs_i32, i32x4, -42i32, 42i32 }
    test_vec_abs! { test_vec_abs_i64, i64x2, -42i64, 42i64 }
    test_vec_abs! { test_vec_abs_f32, f32x4, -42f32, 42f32 }
    test_vec_abs! { test_vec_abs_f64, f64x2, -42f64, 42f64 }

    test_vec_1! { test_vec_nabs, vec_nabs, f32x4,
    [core::f32::consts::PI, 1.0, 0.0, -1.0],
    [-core::f32::consts::PI, -1.0, 0.0, -1.0] }

    test_vec_2! { test_vec_andc, vec_andc, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b10000000],
    [0b11001100, 0b00001100, 0b11000000, 0b01001100] }

    test_vec_2! { test_vec_and, vec_and, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b00000000, 0b11000000, 0b00001100, 0b00000000] }

    test_vec_2! { test_vec_nand, vec_nand, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b00000000, !0b11000000, !0b00001100, !0b00000000] }

    test_vec_2! { test_vec_orc, vec_orc, u32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11001100 | !0b00110011, 0b11001100 | !0b11110011, 0b11001100 | !0b00001100, 0b11001100 | !0b00000000] }

    test_vec_2! { test_vec_or, vec_or, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11111111, 0b11111111, 0b11001100, 0b11001100] }

    test_vec_2! { test_vec_nor, vec_nor, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b11111111, !0b11111111, !0b11001100, !0b11001100] }

    test_vec_2! { test_vec_xor, vec_xor, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [0b11111111, 0b00111111, 0b11000000, 0b11001100] }

    test_vec_2! { test_vec_eqv, vec_eqv, i32x4,
    [0b11001100, 0b11001100, 0b11001100, 0b11001100],
    [0b00110011, 0b11110011, 0b00001100, 0b00000000],
    [!0b11111111, !0b00111111, !0b11000000, !0b11001100] }

    test_vec_1! { test_vec_floor_f32, vec_floor, f32x4,
        [1.1, 1.9, -0.5, -0.9],
        [1.0, 1.0, -1.0, -1.0]
    }

    test_vec_1! { test_vec_floor_f64_1, vec_floor, f64x2,
        [1.1, 1.9],
        [1.0, 1.0]
    }
    test_vec_1! { test_vec_floor_f64_2, vec_floor, f64x2,
        [-0.5, -0.9],
        [-1.0, -1.0]
    }

    test_vec_1! { test_vec_ceil_f32, vec_ceil, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [1.0, 1.0, 1.0, 1.0]
    }
    test_vec_1! { test_vec_ceil_f64_1, vec_ceil, f64x2,
        [0.1, 0.5],
        [1.0, 1.0]
    }
    test_vec_1! { test_vec_ceil_f64_2, vec_ceil, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_1! { test_vec_round_f32, vec_round, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_round_f32_even_odd, vec_round, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_round_f64_1, vec_round, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_round_f64_2, vec_round, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_1! { test_vec_roundc_f32, vec_roundc, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_roundc_f32_even_odd, vec_roundc, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_roundc_f64_1, vec_roundc, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_roundc_f64_2, vec_roundc, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_1! { test_vec_rint_f32, vec_rint, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.0, 0.0, 1.0, 1.0]
    }

    test_vec_1! { test_vec_rint_f32_even_odd, vec_rint, f32x4,
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 2.0, 2.0, 4.0]
    }

    test_vec_1! { test_vec_rint_f64_1, vec_rint, f64x2,
        [0.1, 0.5],
        [0.0, 0.0]
    }
    test_vec_1! { test_vec_rint_f64_2, vec_rint, f64x2,
        [0.6, 0.9],
        [1.0, 1.0]
    }

    test_vec_2! { test_vec_sll, vec_sll, i32x4, u8x16 -> i32x4,
    [1, 1, 1, 1],
    [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 8],
    [1 << 2, 1 << 3, 1 << 4, 1] }

    test_vec_2! { test_vec_srl, vec_srl, i32x4, u8x16 -> i32x4,
    [0b1000, 0b1000, 0b1000, 0b1000],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [4, 2, 1, 8] }

    test_vec_2! { test_vec_sral_pos, vec_sral, u32x4, u8x16 -> i32x4,
    [0b1000, 0b1000, 0b1000, 0b1000],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [4, 2, 1, 8] }

    test_vec_2! { test_vec_sral_neg, vec_sral, i32x4, u8x16 -> i32x4,
    [-8, -8, -8, -8],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 16],
    [-4, -2, -1, -8] }

    test_vec_1! { test_vec_reve_f32, vec_reve, f32x4,
        [0.1, 0.5, 0.6, 0.9],
        [0.9, 0.6, 0.5, 0.1]
    }

    test_vec_1! { test_vec_revb_u32, vec_revb, u32x4,
        [0xAABBCCDD, 0xEEFF0011, 0x22334455, 0x66778899],
        [0xDDCCBBAA, 0x1100FFEE, 0x55443322, 0x99887766]
    }

    test_vec_2! { test_vec_mergeh_u32, vec_mergeh, u32x4,
        [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD],
        [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        [0xAAAAAAAA, 0x00000000, 0xBBBBBBBB, 0x11111111]
    }

    test_vec_2! { test_vec_mergel_u32, vec_mergel, u32x4,
        [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD],
        [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        [0xCCCCCCCC, 0x22222222, 0xDDDDDDDD, 0x33333333]
    }

    macro_rules! test_vec_perm {
        {$name:ident,
         $shorttype:ident, $longtype:ident,
         [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "vector")]
            unsafe fn $name() {
                let a: $longtype = transmute($shorttype::new($($a),+));
                let b: $longtype = transmute($shorttype::new($($b),+));
                let c: vector_unsigned_char = transmute(u8x16::new($($c),+));
                let d = $shorttype::new($($d),+);

                let r: $shorttype = transmute(vec_perm(a, b, c));
                assert_eq!(d, r);
            }
        }
    }

    test_vec_perm! {test_vec_perm_u8x16,
    u8x16, vector_unsigned_char,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105, 6, 7, 106, 107]}
    test_vec_perm! {test_vec_perm_i8x16,
    i8x16, vector_signed_char,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105, 6, 7, 106, 107]}

    test_vec_perm! {test_vec_perm_m8x16,
    m8x16, vector_bool_char,
    [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false],
    [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [false, false, true, true, false, false, true, true, false, false, true, true, false, false, true, true]}
    test_vec_perm! {test_vec_perm_u16x8,
    u16x8, vector_unsigned_short,
    [0, 1, 2, 3, 4, 5, 6, 7],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 10, 1, 11, 2, 12, 3, 13]}
    test_vec_perm! {test_vec_perm_i16x8,
    i16x8, vector_signed_short,
    [0, 1, 2, 3, 4, 5, 6, 7],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [0, 10, 1, 11, 2, 12, 3, 13]}
    test_vec_perm! {test_vec_perm_m16x8,
    m16x8, vector_bool_short,
    [false, false, false, false, false, false, false, false],
    [true, true, true, true, true, true, true, true],
    [0x00, 0x01, 0x10, 0x11, 0x02, 0x03, 0x12, 0x13,
     0x04, 0x05, 0x14, 0x15, 0x06, 0x07, 0x16, 0x17],
    [false, true, false, true, false, true, false, true]}

    test_vec_perm! {test_vec_perm_u32x4,
    u32x4, vector_unsigned_int,
    [0, 1, 2, 3],
    [10, 11, 12, 13],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0, 10, 1, 11]}
    test_vec_perm! {test_vec_perm_i32x4,
    i32x4, vector_signed_int,
    [0, 1, 2, 3],
    [10, 11, 12, 13],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0, 10, 1, 11]}
    test_vec_perm! {test_vec_perm_m32x4,
    m32x4, vector_bool_int,
    [false, false, false, false],
    [true, true, true, true],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [false, true, false, true]}
    test_vec_perm! {test_vec_perm_f32x4,
    f32x4, vector_float,
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 1.1, 1.2, 1.3],
    [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
     0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17],
    [0.0, 1.0, 1.0, 1.1]}

    test_vec_1! { test_vec_sqrt, vec_sqrt, f32x4,
    [core::f32::consts::PI, 1.0, 25.0, 2.0],
    [core::f32::consts::PI.sqrt(), 1.0, 5.0, core::f32::consts::SQRT_2] }

    test_vec_2! { test_vec_find_any_eq, vec_find_any_eq, i32x4, i32x4 -> u32x4,
        [1, -2, 3, -4],
        [-5, 3, -7, 8],
        [0, 0, 0xFFFFFFFF, 0]
    }

    test_vec_2! { test_vec_find_any_ne, vec_find_any_ne, i32x4, i32x4 -> u32x4,
        [1, -2, 3, -4],
        [-5, 3, -7, 8],
        [0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF]
    }

    test_vec_2! { test_vec_find_any_eq_idx_1, vec_find_any_eq_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [5, 3, 7, 8],
        [0, 8, 0, 0]
    }
    test_vec_2! { test_vec_find_any_eq_idx_2, vec_find_any_eq_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 16, 0, 0]
    }

    test_vec_2! { test_vec_find_any_ne_idx_1, vec_find_any_ne_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [1, 5, 3, 4],
        [0, 4, 0, 0]
    }
    test_vec_2! { test_vec_find_any_ne_idx_2, vec_find_any_ne_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [0, 16, 0, 0]
    }

    test_vec_2! { test_vec_find_any_eq_or_0_idx_1, vec_find_any_eq_or_0_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 0, 4],
        [5, 6, 7, 8],
        [0, 8, 0, 0]
    }
    test_vec_2! { test_vec_find_any_ne_or_0_idx_1, vec_find_any_ne_or_0_idx, i32x4, i32x4 -> u32x4,
        [1, 2, 0, 4],
        [1, 2, 3, 4],
        [0, 8, 0, 0]
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_cc() {
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let (d, c) = unsafe { vec_find_any_eq_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 0, -1, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let (d, c) = unsafe { vec_find_any_eq_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_cc() {
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let (d, c) = unsafe { vec_find_any_ne_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[-1, -1, 0, -1]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let (d, c) = unsafe { vec_find_any_ne_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_idx_cc() {
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let (d, c) = unsafe { vec_find_any_eq_idx_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 8, 0, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let (d, c) = unsafe { vec_find_any_eq_idx_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_idx_cc() {
        let a = vector_unsigned_int([5, 2, 3, 4]);
        let b = vector_unsigned_int([5, 3, 7, 8]);

        let (d, c) = unsafe { vec_find_any_ne_idx_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let (d, c) = unsafe { vec_find_any_ne_idx_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_eq_or_0_idx_cc() {
        // if no element of a matches any element of b with an equal value, and there is at least one element from a with a value of 0
        let a = vector_unsigned_int([0, 1, 2, 3]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let (d, c) = unsafe { vec_find_any_eq_or_0_idx_cc(a, b) };
        assert_eq!(c, 0);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if at least one element of a matches any element of b with an equal value, and no elements of a with a value of 0
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 2, 3, 4]);
        let (d, c) = unsafe { vec_find_any_eq_or_0_idx_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        // if at least one element of a matches any element of b with an equal value, and there is at least one element from a has a value of 0
        let a = vector_unsigned_int([1, 2, 3, 0]);
        let b = vector_unsigned_int([1, 2, 3, 4]);
        let (d, c) = unsafe { vec_find_any_eq_or_0_idx_cc(a, b) };
        assert_eq!(c, 2);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if no element of a matches any element of b with an equal value, and there is no element from a with a value of 0.
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let (d, c) = unsafe { vec_find_any_eq_or_0_idx_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_find_any_ne_or_0_idx_cc() {
        // if no element of a matches any element of b with a not equal value, and there is at least one element from a with a value of 0.
        let a = vector_unsigned_int([0, 1, 2, 3]);
        let b = vector_unsigned_int([4, 1, 2, 3]);
        let (d, c) = unsafe { vec_find_any_ne_or_0_idx_cc(a, b) };
        assert_eq!(c, 0);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if at least one element of a matches any element of b with a not equal value, and no elements of a with a value of 0.
        let a = vector_unsigned_int([4, 2, 3, 4]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let (d, c) = unsafe { vec_find_any_ne_or_0_idx_cc(a, b) };
        assert_eq!(c, 1);
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);

        // if at least one element of a matches any element of b with a not equal value, and there is at least one element from a has a value of 0.
        let a = vector_unsigned_int([1, 0, 1, 1]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let (d, c) = unsafe { vec_find_any_ne_or_0_idx_cc(a, b) };
        assert_eq!(c, 2);
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);

        // if no element of a matches any element of b with a not equal value, and there is no element from a with a value of 0.
        let a = vector_unsigned_int([4, 4, 4, 4]);
        let b = vector_unsigned_int([4, 5, 6, 7]);
        let (d, c) = unsafe { vec_find_any_ne_or_0_idx_cc(a, b) };
        assert_eq!(c, 3);
        assert_eq!(d.as_array(), &[0, 16, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vector_load() {
        let expected = [0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC_CCCC, 0xDDDD_DDDD];

        let source: [u32; 8] = [
            0xAAAA_AAAA,
            0xBBBB_BBBB,
            0xCCCC_CCCC,
            0xDDDD_DDDD,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(
            unsafe { vec_xl::<vector_unsigned_int>(0, source.as_ptr()) }.as_array(),
            &expected
        );

        // offset is in bytes
        let source: [u32; 8] = [
            0x0000_AAAA,
            0xAAAA_BBBB,
            0xBBBB_CCCC,
            0xCCCC_DDDD,
            0xDDDD_0000,
            0,
            0,
            0,
        ];
        assert_eq!(
            unsafe { vec_xl::<vector_unsigned_int>(2, source.as_ptr()) }.as_array(),
            &expected
        );
    }

    #[simd_test(enable = "vector")]
    fn test_vector_store() {
        let vec = vector_unsigned_int([0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC_CCCC, 0xDDDD_DDDD]);

        let mut dest = [0u32; 8];
        unsafe { vec_xst(vec, 0, dest.as_mut_ptr()) };
        assert_eq!(
            dest,
            [
                0xAAAA_AAAA,
                0xBBBB_BBBB,
                0xCCCC_CCCC,
                0xDDDD_DDDD,
                0,
                0,
                0,
                0
            ]
        );

        // offset is in bytes
        let mut dest = [0u32; 8];
        unsafe { vec_xst(vec, 2, dest.as_mut_ptr()) };
        assert_eq!(
            dest,
            [
                0x0000_AAAA,
                0xAAAA_BBBB,
                0xBBBB_CCCC,
                0xCCCC_DDDD,
                0xDDDD_0000,
                0,
                0,
                0,
            ]
        );
    }

    #[simd_test(enable = "vector")]
    fn test_vector_lcbb() {
        #[repr(align(64))]
        struct Align64<T>(T);

        static ARRAY: Align64<[u8; 128]> = Align64([0; 128]);

        assert_eq!(unsafe { __lcbb::<64>(ARRAY.0[64..].as_ptr()) }, 16);
        assert_eq!(unsafe { __lcbb::<64>(ARRAY.0[63..].as_ptr()) }, 1);
        assert_eq!(unsafe { __lcbb::<64>(ARRAY.0[56..].as_ptr()) }, 8);
        assert_eq!(unsafe { __lcbb::<64>(ARRAY.0[48..].as_ptr()) }, 16);
    }

    test_vec_2! { test_vec_pack, vec_pack, i16x8, i16x8 -> i8x16,
        [0, 1, -1, 42, 32767, -32768, 30000, -30000],
        [32767, -32768, 12345, -12345, 0, 1, -1, 42],
        [0, 1, -1, 42, -1, 0, 48, -48, -1, 0, 57, -57, 0, 1, -1, 42]
    }

    test_vec_2! { test_vec_packs, vec_packs, i16x8, i16x8 -> i8x16,
        [0, 1, -1, 42, 32767, -32768, 30000, -30000],
        [32767, -32768, 12345, -12345, 0, 1, -1, 42],
        [0, 1, -1, 42, 127, -128, 127, -128, 127, -128, 127, -128, 0, 1, -1, 42]
    }

    test_vec_2! { test_vec_packsu_signed, vec_packsu, i16x8, i16x8 -> u8x16,
        [0, 1, -1, 42, 32767, -32768, 30000, -30000],
        [32767, -32768, 12345, -12345, 0, 1, -1, 42],
        [0, 1, 0, 42, 255, 0, 255, 0, 255, 0, 255, 0, 0, 1, 0, 42]
    }

    test_vec_2! { test_vec_packsu_unsigned, vec_packsu, u16x8, u16x8 -> u8x16,
        [65535, 32768, 1234, 5678, 16, 8, 4, 2],
        [30000, 25000, 20000, 15000, 31, 63, 127, 255],
        [255, 255, 255, 255, 16, 8, 4, 2, 255, 255, 255, 255, 31, 63, 127, 255]
    }

    test_vec_2! { test_vec_rl, vec_rl, u32x4,
        [0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0x12345678],
        [4, 8, 12, 68],
        [0x23456781, 0xBCDEF09A, 0xF0F0F0F0, 0x23456781]
    }

    test_vec_1! { test_vec_unpackh_i, vec_unpackh, i16x8 -> i32x4,
        [0x1234, -2, 0x0F0F, -32768, 0, 0, 0, 0],
        [0x1234, -2, 0x0F0F, -32768]
    }

    test_vec_1! { test_vec_unpackh_u, vec_unpackh, u16x8 -> u32x4,
        [0x1234, 0xFFFF, 0x0F0F, 0x8000, 0, 0, 0, 0],
        [0x1234, 0xFFFF, 0x0F0F, 0x8000]
    }

    test_vec_1! { test_vec_unpackl_i, vec_unpackl, i16x8 -> i32x4,
        [0, 0, 0, 0, 0x1234, -2, 0x0F0F, -32768],
        [0x1234, -2, 0x0F0F, -32768]
    }

    test_vec_1! { test_vec_unpackl_u, vec_unpackl, u16x8 -> u32x4,
        [0, 0, 0, 0, 0x1234, 0xFFFF, 0x0F0F, 0x8000],
        [0x1234, 0xFFFF, 0x0F0F, 0x8000]
    }

    test_vec_2! { test_vec_avg, vec_avg, u32x4,
        [2, 1, u32::MAX, 0],
        [4, 2, 2, 0],
        [3, (1u32 + 2).div_ceil(2), (u32::MAX as u64 + 2u64).div_ceil(2) as u32, 0]
    }

    test_vec_2! { test_vec_checksum, vec_checksum, u32x4,
        [1, 2, 3, u32::MAX],
        [5, 6, 7, 8],
        [0, 12, 0, 0]
    }

    test_vec_2! { test_vec_add_u128, vec_add_u128, u8x16,
        [0x01, 0x05, 0x0F, 0x1A, 0x2F, 0x3F, 0x50, 0x65,
                              0x7A, 0x8F, 0x9A, 0xAD, 0xB0, 0xC3, 0xD5, 0xE8],
        [0xF0, 0xEF, 0xC3, 0xB1, 0x92, 0x71, 0x5A, 0x43,
                              0x3B, 0x29, 0x13, 0x04, 0xD7, 0xA1, 0x8C, 0x76],
        [0xF1, 0xF4, 0xD2, 0xCB, 0xC1, 0xB0, 0xAA, 0xA8, 0xB5, 0xB8, 0xAD, 0xB2, 0x88, 0x65, 0x62, 0x5E]
    }

    #[simd_test(enable = "vector")]
    fn test_vec_addc_u128() {
        unsafe {
            let a = u128::MAX;
            let b = 1u128;

            let d: u128 = transmute(vec_addc_u128(transmute(a), transmute(b)));
            assert!(a.checked_add(b).is_none());
            assert_eq!(d, 1);

            let a = 1u128;
            let b = 1u128;

            let d: u128 = transmute(vec_addc_u128(transmute(a), transmute(b)));
            assert!(a.checked_add(b).is_some());
            assert_eq!(d, 0);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_subc_u128() {
        unsafe {
            let a = 0u128;
            let b = 1u128;

            let d: u128 = transmute(vec_subc_u128(transmute(a), transmute(b)));
            assert!(a.checked_sub(b).is_none());
            assert_eq!(d, 0);

            let a = 1u128;
            let b = 1u128;

            let d: u128 = transmute(vec_subc_u128(transmute(a), transmute(b)));
            assert!(a.checked_sub(b).is_some());
            assert_eq!(d, 1);
        }
    }

    test_vec_2! { test_vec_mule_u, vec_mule, u16x8, u16x8 -> u32x4,
        [0xFFFF, 0, 2, 0, 2, 0, 1, 0],
        [0xFFFF, 0, 4, 0, 0xFFFF, 0, 2, 0],
        [0xFFFE_0001, 8, 0x0001_FFFE, 2]
    }

    test_vec_2! { test_vec_mule_i, vec_mule, i16x8, i16x8 -> i32x4,
        [i16::MIN, 0, -2, 0, 2, 0, 1, 0],
        [i16::MIN, 0, 4, 0, i16::MAX, 0, 2, 0],
        [0x4000_0000, -8, 0xFFFE, 2]
    }

    test_vec_2! { test_vec_mulo_u, vec_mulo, u16x8, u16x8 -> u32x4,
        [0, 0xFFFF, 0, 2, 0, 2, 0, 1],
        [0, 0xFFFF, 0, 4, 0, 0xFFFF, 0, 2],
        [0xFFFE_0001, 8, 0x0001_FFFE, 2]
    }

    test_vec_2! { test_vec_mulo_i, vec_mulo, i16x8, i16x8 -> i32x4,
        [0, i16::MIN, 0, -2, 0, 2, 0, 1],
        [0, i16::MIN, 0, 4, 0, i16::MAX, 0, 2],
        [0x4000_0000, -8, 0xFFFE, 2]
    }

    test_vec_2! { test_vec_mulh_u, vec_mulh, u32x4, u32x4 -> u32x4,
        [u32::MAX, 2, 2, 1],
        [u32::MAX, 4, u32::MAX, 2],
        [u32::MAX - 1, 0, 1, 0]
    }

    test_vec_2! { test_vec_mulh_i, vec_mulh, i32x4, i32x4 -> i32x4,
        [i32::MIN, -2, 2, 1],
        [i32::MIN, 4, i32::MAX, 2],
        [0x4000_0000, -1, 0, 0]
    }

    test_vec_2! { test_vec_gfmsum_1, vec_gfmsum, u16x8, u16x8 -> u32x4,
        [0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1357, 0x2468, 0xACE0, 0xBDF0],
        [0xFFFF, 0x0001, 0x8000, 0x7FFF, 0xAAAA, 0x5555, 0x1234, 0x5678],
        [0xE13A794, 0x68764A50, 0x94AA3E, 0x2C93F300]
    }

    test_vec_2! { test_vec_gfmsum_2, vec_gfmsum, u16x8, u16x8 -> u32x4,
        [0x0000, 0xFFFF, 0xAAAA, 0x5555, 0x1234, 0x5678, 0x9ABC, 0xDEF0],
        [0xFFFF, 0x0000, 0x5555, 0xAAAA, 0x0001, 0x8000, 0x7FFF, 0x1357],
        [0, 0, 0x2B3C1234, 0x3781D244]
    }

    #[simd_test(enable = "vector")]
    fn test_vec_gfmsum_128() {
        let a = vector_unsigned_long_long([1, 2]);
        let b = vector_unsigned_long_long([3, 4]);

        let d: u128 = unsafe { transmute(vec_gfmsum_128(a, b)) };
        assert_eq!(d, 11);

        let a = vector_unsigned_long_long([0x0101010101010101, 0x0202020202020202]);
        let b = vector_unsigned_long_long([0x0404040404040404, 0x0505050505050505]);

        let d: u128 = unsafe { transmute(vec_gfmsum_128(a, b)) };
        assert_eq!(d, 0xE000E000E000E000E000E000E000E);
    }

    #[simd_test(enable = "vector-enhancements-1")]
    fn test_vec_bperm_u128() {
        let a = vector_unsigned_char([65, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let b = vector_unsigned_char([
            0, 0, 0, 0, 1, 1, 1, 1, 128, 128, 128, 128, 255, 255, 255, 255,
        ]);
        let d = unsafe { vec_bperm_u128(a, b) };
        assert_eq!(d.as_array(), &[0xF00, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_sel() {
        let a = vector_signed_int([1, 2, 3, 4]);
        let b = vector_signed_int([5, 6, 7, 8]);

        let e = vector_unsigned_int([9, 10, 11, 12]);
        let f = vector_unsigned_int([9, 9, 11, 11]);

        let c: vector_bool_int = unsafe { simd_eq(e, f) };
        assert_eq!(c.as_array(), &[!0, 0, !0, 0]);
        let d: vector_signed_int = unsafe { vec_sel(a, b, c) };
        assert_eq!(d.as_array(), &[5, 2, 7, 4]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_gather_element() {
        let a1: [u32; 10] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        let a2: [u32; 10] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29];

        let v1 = vector_unsigned_int([1, 2, 3, 4]);
        let v2 = vector_unsigned_int([1, 2, 3, 4]);

        let sizeof_int = core::mem::size_of::<u32>() as u32;
        let v3 = vector_unsigned_int([
            5 * sizeof_int,
            8 * sizeof_int,
            9 * sizeof_int,
            6 * sizeof_int,
        ]);

        unsafe {
            let d1 = vec_gather_element::<_, 0>(v1, v3, a1.as_ptr());
            assert_eq!(d1.as_array(), &[15, 2, 3, 4]);
            let d2 = vec_gather_element::<_, 0>(v2, v3, a2.as_ptr());
            assert_eq!(d2.as_array(), &[25, 2, 3, 4]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_fp_test_data_class() {
        let mut cc = 42;

        let v1 = vector_double([0.0, f64::NAN]);
        let v2 = vector_double([f64::INFINITY, 1.0]);
        let v3 = vector_double([1.0, 2.0]);

        unsafe {
            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_ZERO>(v1, &mut cc);
            assert_eq!(cc, 1);
            assert_eq!(d.as_array(), &[!0, 0]);

            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_NAN>(v1, &mut cc);
            assert_eq!(cc, 1);
            assert_eq!(d.as_array(), &[0, !0]);

            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_INFINITY>(v2, &mut cc);
            assert_eq!(cc, 1);
            assert_eq!(d.as_array(), &[!0, 0]);

            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_INFINITY_N>(v2, &mut cc);
            assert_eq!(cc, 3);
            assert_eq!(d.as_array(), &[0, 0]);

            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_NORMAL>(v2, &mut cc);
            assert_eq!(cc, 1);
            assert_eq!(d.as_array(), &[0, !0]);

            let d = vec_fp_test_data_class::<_, __VEC_CLASS_FP_NORMAL>(v3, &mut cc);
            assert_eq!(cc, 0);
            assert_eq!(d.as_array(), &[!0, !0]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_fp_any_all_nan_numeric() {
        unsafe {
            assert_eq!(
                vec_all_nan(vector_double([f64::NAN, f64::NAN])),
                i32::from(true)
            );
            assert_eq!(
                vec_all_nan(vector_double([f64::NAN, 1.0])),
                i32::from(false)
            );
            assert_eq!(vec_all_nan(vector_double([0.0, 1.0])), i32::from(false));

            assert_eq!(
                vec_any_nan(vector_double([f64::NAN, f64::NAN])),
                i32::from(true)
            );
            assert_eq!(vec_any_nan(vector_double([f64::NAN, 1.0])), i32::from(true));
            assert_eq!(vec_any_nan(vector_double([0.0, 1.0])), i32::from(false));

            assert_eq!(
                vec_all_numeric(vector_double([f64::NAN, f64::NAN])),
                i32::from(false)
            );
            assert_eq!(
                vec_all_numeric(vector_double([f64::NAN, 1.0])),
                i32::from(false)
            );
            assert_eq!(vec_all_numeric(vector_double([0.0, 1.0])), i32::from(true));

            assert_eq!(
                vec_any_numeric(vector_double([f64::NAN, f64::NAN])),
                i32::from(false)
            );
            assert_eq!(
                vec_any_numeric(vector_double([f64::NAN, 1.0])),
                i32::from(true)
            );
            assert_eq!(vec_any_numeric(vector_double([0.0, 1.0])), i32::from(true));

            // "numeric" means "not NaN". infinities are numeric
            assert_eq!(
                vec_all_numeric(vector_double([f64::INFINITY, f64::NEG_INFINITY])),
                i32::from(true)
            );
            assert_eq!(
                vec_any_numeric(vector_double([f64::INFINITY, f64::NEG_INFINITY])),
                i32::from(true)
            );
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_test_mask() {
        unsafe {
            let v = vector_unsigned_long_long([0xFF00FF00FF00FF00; 2]);
            let m = vector_unsigned_long_long([0x0000FF000000FF00; 2]);
            assert_eq!(vec_test_mask(v, m), 3);

            let v = vector_unsigned_long_long([u64::MAX; 2]);
            let m = vector_unsigned_long_long([0; 2]);
            assert_eq!(vec_test_mask(v, m), 0);

            let v = vector_unsigned_long_long([0; 2]);
            let m = vector_unsigned_long_long([u64::MAX; 2]);
            assert_eq!(vec_test_mask(v, m), 0);

            let v = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA; 2]);
            let m = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA; 2]);
            assert_eq!(vec_test_mask(v, m), 3);
        }
    }

    #[simd_test(enable = "vector-enhancements-2")]
    fn test_vec_search_string_cc() {
        unsafe {
            let b = vector_unsigned_char(*b"ABCD------------");
            let c = vector_unsigned_char([4; 16]);

            let haystack = vector_unsigned_char(*b"__ABCD__________");
            let (result, d) = vec_search_string_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 2);
            assert_eq!(d, 2);

            let haystack = vector_unsigned_char(*b"___ABCD_________");
            let (result, d) = vec_search_string_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 3);
            assert_eq!(d, 2);

            let haystack = vector_unsigned_char(*b"________________");
            let (result, d) = vec_search_string_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 16);
            assert_eq!(d, 0);

            let haystack = vector_unsigned_char(*b"______\0_________");
            let (result, d) = vec_search_string_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 16);
            assert_eq!(d, 0);

            let haystack = vector_unsigned_char(*b"______\0__ABCD___");
            let (result, d) = vec_search_string_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 9);
            assert_eq!(d, 2);
        }
    }

    #[simd_test(enable = "vector-enhancements-2")]
    fn test_vec_search_string_until_zero_cc() {
        unsafe {
            let b = vector_unsigned_char(*b"ABCD\0\0\0\0\0\0\0\0\0\0\0\0");
            let c = vector_unsigned_char([16; 16]);

            let haystack = vector_unsigned_char(*b"__ABCD__________");
            let (result, d) = vec_search_string_until_zero_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 2);
            assert_eq!(d, 2);

            let haystack = vector_unsigned_char(*b"___ABCD_________");
            let (result, d) = vec_search_string_until_zero_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 3);
            assert_eq!(d, 2);

            let haystack = vector_unsigned_char(*b"________________");
            let (result, d) = vec_search_string_until_zero_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 16);
            assert_eq!(d, 0);

            let haystack = vector_unsigned_char(*b"______\0_________");
            let (result, d) = vec_search_string_until_zero_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 16);
            assert_eq!(d, 1);

            let haystack = vector_unsigned_char(*b"______\0__ABCD___");
            let (result, d) = vec_search_string_until_zero_cc(haystack, b, c);
            assert_eq!(result.as_array()[7], 16);
            assert_eq!(d, 1);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_doublee() {
        unsafe {
            let v = vector_float([1.0, 2.0, 3.0, 4.0]);
            assert_eq!(vec_doublee(v).as_array(), &[1.0, 3.0]);

            let v = vector_float([f32::NAN, 2.0, f32::INFINITY, 4.0]);
            let d = vec_doublee(v);
            assert!(d.as_array()[0].is_nan());
            assert_eq!(d.as_array()[1], f64::INFINITY);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_floate() {
        // NOTE: indices 1 and 3 can have an arbitrary value. With the C version
        // these are poison values, our version initializes the memory but its
        // value still should not be relied upon by application code.
        unsafe {
            let v = vector_double([1.0, 2.0]);
            let d = vec_floate(v);
            assert_eq!(d.as_array()[0], 1.0);
            assert_eq!(d.as_array()[2], 2.0);

            let v = vector_double([f64::NAN, f64::INFINITY]);
            let d = vec_floate(v);
            assert!(d.as_array()[0].is_nan());
            assert_eq!(d.as_array()[2], f32::INFINITY);

            let v = vector_double([f64::MIN, f64::MAX]);
            let d = vec_floate(v);
            assert_eq!(d.as_array()[0], f64::MIN as f32);
            assert_eq!(d.as_array()[2], f64::MAX as f32);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_extend_s64() {
        unsafe {
            let v = vector_signed_char([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            assert_eq!(vec_extend_s64(v).as_array(), &[7, 15]);

            let v = vector_signed_short([0, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(vec_extend_s64(v).as_array(), &[3, 7]);

            let v = vector_signed_int([0, 1, 2, 3]);
            assert_eq!(vec_extend_s64(v).as_array(), &[1, 3]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_signed() {
        unsafe {
            let v = vector_float([1.0, 2.5, -2.5, -0.0]);
            assert_eq!(vec_signed(v).as_array(), &[1, 2, -2, 0]);

            let v = vector_double([2.5, -2.5]);
            assert_eq!(vec_signed(v).as_array(), &[2, -2]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_unsigned() {
        // NOTE: converting a negative floating point value is UB!
        unsafe {
            let v = vector_float([1.0, 2.5, 3.5, 0.0]);
            assert_eq!(vec_unsigned(v).as_array(), &[1, 2, 3, 0]);

            let v = vector_double([2.5, 3.5]);
            assert_eq!(vec_unsigned(v).as_array(), &[2, 3]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cp_until_zero() {
        unsafe {
            let v = vector_signed_int([1, 2, 3, 4]);
            let d = vec_cp_until_zero(v);
            assert_eq!(d.as_array(), &[1, 2, 3, 4]);

            let v = vector_signed_int([1, 2, 0, 4]);
            let d = vec_cp_until_zero(v);
            assert_eq!(d.as_array(), &[1, 2, 0, 0]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cp_until_zero_cc() {
        unsafe {
            let v = vector_signed_int([1, 2, 3, 4]);
            let (d, cc) = vec_cp_until_zero_cc(v);
            assert_eq!(d.as_array(), &[1, 2, 3, 4]);
            assert_eq!(cc, 3);

            let v = vector_signed_int([1, 2, 0, 4]);
            let (d, cc) = vec_cp_until_zero_cc(v);
            assert_eq!(d.as_array(), &[1, 2, 0, 0]);
            assert_eq!(cc, 0);
        }
    }

    #[simd_test(enable = "vector-enhancements-1")]
    fn test_vec_msum_u128() {
        let a = vector_unsigned_long_long([1, 2]);
        let b = vector_unsigned_long_long([3, 4]);

        unsafe {
            let c: vector_unsigned_char = transmute(100u128);

            let d: u128 = transmute(vec_msum_u128::<0>(a, b, c));
            assert_eq!(d, (1 * 3) + (2 * 4) + 100);

            let d: u128 = transmute(vec_msum_u128::<4>(a, b, c));
            assert_eq!(d, (1 * 3) + (2 * 4) * 2 + 100);

            let d: u128 = transmute(vec_msum_u128::<8>(a, b, c));
            assert_eq!(d, (1 * 3) * 2 + (2 * 4) + 100);

            let d: u128 = transmute(vec_msum_u128::<12>(a, b, c));
            assert_eq!(d, (1 * 3) * 2 + (2 * 4) * 2 + 100);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_sld() {
        let a = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA]);
        let b = vector_unsigned_long_long([0xBBBBBBBBBBBBBBBB, 0xBBBBBBBBBBBBBBBB]);

        unsafe {
            let d = vec_sld::<_, 4>(a, b);
            assert_eq!(d.as_array(), &[0xAAAAAAAAAAAAAAAA, 0xAAAAAAAABBBBBBBB]);
        }
    }

    #[simd_test(enable = "vector")]
    fn test_vec_sldw() {
        let a = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA]);
        let b = vector_unsigned_long_long([0xBBBBBBBBBBBBBBBB, 0xBBBBBBBBBBBBBBBB]);

        unsafe {
            let d = vec_sldw::<_, 1>(a, b);
            assert_eq!(d.as_array(), &[0xAAAAAAAAAAAAAAAA, 0xAAAAAAAABBBBBBBB]);
        }
    }

    #[simd_test(enable = "vector-enhancements-2")]
    fn test_vec_sldb() {
        let a = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA]);
        let b = vector_unsigned_long_long([0xBBBBBBBBBBBBBBBB, 0xBBBBBBBBBBBBBBBB]);

        unsafe {
            let d = vec_sldb::<_, 4>(a, b);
            assert_eq!(d.as_array(), &[0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAB]);
        }
    }

    #[simd_test(enable = "vector-enhancements-2")]
    fn test_vec_srdb() {
        let a = vector_unsigned_long_long([0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA]);
        let b = vector_unsigned_long_long([0xBBBBBBBBBBBBBBBB, 0xBBBBBBBBBBBBBBBB]);

        unsafe {
            let d = vec_srdb::<_, 4>(a, b);
            assert_eq!(d.as_array(), &[0xABBBBBBBBBBBBBBB, 0xBBBBBBBBBBBBBBBB]);
        }
    }

    const GT: u32 = 0x20000000;
    const LT: u32 = 0x40000000;
    const EQ: u32 = 0x80000000;

    #[simd_test(enable = "vector")]
    fn test_vec_cmprg() {
        let a = vector_unsigned_int([11, 22, 33, 44]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmprg(a, b, c) };
        assert_eq!(d.as_array(), &[!0, 0, !0, 0]);

        let c = vector_unsigned_int([GT, LT, 0, 0]);
        let d = unsafe { vec_cmprg(a, b, c) };
        assert_eq!(d.as_array(), &[!0, 0, 0, 0]);

        let a = vector_unsigned_int([11, 22, 33, 30]);
        let b = vector_unsigned_int([10, 20, 30, 30]);

        let c = vector_unsigned_int([GT, LT, EQ, EQ]);
        let d = unsafe { vec_cmprg(a, b, c) };
        assert_eq!(d.as_array(), &[!0, 0, 0, !0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cmpnrg() {
        let a = vector_unsigned_int([11, 22, 33, 44]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmpnrg(a, b, c) };
        assert_eq!(d.as_array(), &[0, !0, 0, !0]);

        let c = vector_unsigned_int([GT, LT, 0, 0]);
        let d = unsafe { vec_cmpnrg(a, b, c) };
        assert_eq!(d.as_array(), &[0, !0, !0, !0]);

        let a = vector_unsigned_int([11, 22, 33, 30]);
        let b = vector_unsigned_int([10, 20, 30, 30]);

        let c = vector_unsigned_int([GT, LT, EQ, EQ]);
        let d = unsafe { vec_cmpnrg(a, b, c) };
        assert_eq!(d.as_array(), &[0, !0, !0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cmprg_idx() {
        let a = vector_unsigned_int([1, 11, 22, 33]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmprg_idx(a, b, c) };
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cmpnrg_idx() {
        let a = vector_unsigned_int([1, 11, 22, 33]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmpnrg_idx(a, b, c) };
        assert_eq!(d.as_array(), &[0, 0, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cmprg_or_0_idx() {
        let a = vector_unsigned_int([1, 0, 22, 33]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmprg_or_0_idx(a, b, c) };
        assert_eq!(d.as_array(), &[0, 4, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_cmpnrg_or_0_idx() {
        let a = vector_unsigned_int([11, 33, 0, 22]);
        let b = vector_unsigned_int([10, 20, 30, 40]);

        let c = vector_unsigned_int([GT, LT, GT, LT]);
        let d = unsafe { vec_cmpnrg_or_0_idx(a, b, c) };
        assert_eq!(d.as_array(), &[0, 8, 0, 0]);
    }

    test_vec_2! { test_vec_cmpgt, vec_cmpgt, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 3.14],
        [2.0, f32::NAN, 5.0, 2.0],
        [0, 0, 0, !0]
    }

    test_vec_2! { test_vec_cmpge, vec_cmpge, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 3.14],
        [1.0, f32::NAN, 5.0, 2.0],
        [!0, 0, 0, !0]
    }

    test_vec_2! { test_vec_cmplt, vec_cmplt, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 2.0],
        [2.0, f32::NAN, 5.0, 2.0],
        [!0, 0, 0, 0]
    }

    test_vec_2! { test_vec_cmple, vec_cmple, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 2.0],
        [1.0, f32::NAN, 5.0, 3.14],
        [!0, 0, 0, !0]
    }

    test_vec_2! { test_vec_cmpeq, vec_cmpeq, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 2.0],
        [1.0, f32::NAN, 5.0, 3.14],
        [!0, 0, 0, 0]
    }

    test_vec_2! { test_vec_cmpne, vec_cmpne, f32x4, f32x4 -> i32x4,
        [1.0, f32::NAN, f32::NAN, 2.0],
        [1.0, f32::NAN, 5.0, 3.14],
        [0, !0, !0, !0]
    }

    #[simd_test(enable = "vector")]
    fn test_vec_meadd() {
        let a = vector_unsigned_short([1, 0, 2, 0, 3, 0, 4, 0]);
        let b = vector_unsigned_short([5, 0, 6, 0, 7, 0, 8, 0]);
        let c = vector_unsigned_int([2, 2, 2, 2]);

        let d = unsafe { vec_meadd(a, b, c) };
        assert_eq!(d.as_array(), &[7, 14, 23, 34]);

        let a = vector_signed_short([1, 0, 2, 0, 3, 0, 4, 0]);
        let b = vector_signed_short([5, 0, 6, 0, 7, 0, 8, 0]);
        let c = vector_signed_int([2, -2, 2, -2]);

        let d = unsafe { vec_meadd(a, b, c) };
        assert_eq!(d.as_array(), &[7, 10, 23, 30]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_moadd() {
        let a = vector_unsigned_short([0, 1, 0, 2, 0, 3, 0, 4]);
        let b = vector_unsigned_short([0, 5, 0, 6, 0, 7, 0, 8]);
        let c = vector_unsigned_int([2, 2, 2, 2]);

        let d = unsafe { vec_moadd(a, b, c) };
        assert_eq!(d.as_array(), &[7, 14, 23, 34]);

        let a = vector_signed_short([0, 1, 0, 2, 0, 3, 0, 4]);
        let b = vector_signed_short([0, 5, 0, 6, 0, 7, 0, 8]);
        let c = vector_signed_int([2, -2, 2, -2]);

        let d = unsafe { vec_moadd(a, b, c) };
        assert_eq!(d.as_array(), &[7, 10, 23, 30]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_mhadd() {
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let c = vector_unsigned_int([u32::MAX; 4]);

        let d = unsafe { vec_mhadd(a, b, c) };
        assert_eq!(d.as_array(), &[1, 1, 1, 1]);

        let a = vector_signed_int([-1, -2, -3, -4]);
        let b = vector_signed_int([5, 6, 7, 8]);
        let c = vector_signed_int([i32::MIN; 4]);

        let d = unsafe { vec_mhadd(a, b, c) };
        assert_eq!(d.as_array(), &[-1, -1, -1, -1]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_mladd() {
        let a = vector_unsigned_int([1, 2, 3, 4]);
        let b = vector_unsigned_int([5, 6, 7, 8]);
        let c = vector_unsigned_int([2, 2, 2, 2]);

        let d = unsafe { vec_mladd(a, b, c) };
        assert_eq!(d.as_array(), &[7, 14, 23, 34]);

        let a = vector_signed_int([-1, -2, -3, -4]);
        let b = vector_signed_int([5, 6, 7, 8]);
        let c = vector_signed_int([2, 2, 2, 2]);

        let d = unsafe { vec_mladd(a, b, c) };
        assert_eq!(d.as_array(), &[-3, -10, -19, -30]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_extract() {
        let v = vector_unsigned_int([1, 2, 3, 4]);

        assert_eq!(unsafe { vec_extract(v, 1) }, 2);
        assert_eq!(unsafe { vec_extract(v, 4 + 2) }, 3);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_insert() {
        let mut v = vector_unsigned_int([1, 2, 3, 4]);

        v = unsafe { vec_insert(42, v, 1) };
        assert_eq!(v.as_array(), &[1, 42, 3, 4]);

        v = unsafe { vec_insert(64, v, 6) };
        assert_eq!(v.as_array(), &[1, 42, 64, 4]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_promote() {
        let v: vector_unsigned_int = unsafe { vec_promote(42, 1).assume_init() };
        assert_eq!(v.as_array(), &[0, 42, 0, 0]);
    }

    #[simd_test(enable = "vector")]
    fn test_vec_insert_and_zero() {
        let v = unsafe { vec_insert_and_zero::<vector_unsigned_int>(&42u32) };
        assert_eq!(v.as_array(), vector_unsigned_int([0, 42, 0, 0]).as_array());
    }
}
