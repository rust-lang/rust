// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-loongarch/lasx.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasx.spec
// ```

use super::types::*;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.lasx.xvsll.b"]
    fn __lasx_xvsll_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsll.h"]
    fn __lasx_xvsll_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsll.w"]
    fn __lasx_xvsll_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsll.d"]
    fn __lasx_xvsll_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslli.b"]
    fn __lasx_xvslli_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslli.h"]
    fn __lasx_xvslli_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslli.w"]
    fn __lasx_xvslli_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslli.d"]
    fn __lasx_xvslli_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsra.b"]
    fn __lasx_xvsra_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsra.h"]
    fn __lasx_xvsra_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsra.w"]
    fn __lasx_xvsra_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsra.d"]
    fn __lasx_xvsra_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrai.b"]
    fn __lasx_xvsrai_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrai.h"]
    fn __lasx_xvsrai_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrai.w"]
    fn __lasx_xvsrai_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrai.d"]
    fn __lasx_xvsrai_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrar.b"]
    fn __lasx_xvsrar_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrar.h"]
    fn __lasx_xvsrar_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrar.w"]
    fn __lasx_xvsrar_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrar.d"]
    fn __lasx_xvsrar_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrari.b"]
    fn __lasx_xvsrari_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrari.h"]
    fn __lasx_xvsrari_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrari.w"]
    fn __lasx_xvsrari_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrari.d"]
    fn __lasx_xvsrari_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrl.b"]
    fn __lasx_xvsrl_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrl.h"]
    fn __lasx_xvsrl_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrl.w"]
    fn __lasx_xvsrl_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrl.d"]
    fn __lasx_xvsrl_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrli.b"]
    fn __lasx_xvsrli_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrli.h"]
    fn __lasx_xvsrli_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrli.w"]
    fn __lasx_xvsrli_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrli.d"]
    fn __lasx_xvsrli_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.b"]
    fn __lasx_xvsrlr_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.h"]
    fn __lasx_xvsrlr_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.w"]
    fn __lasx_xvsrlr_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.d"]
    fn __lasx_xvsrlr_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.b"]
    fn __lasx_xvsrlri_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.h"]
    fn __lasx_xvsrlri_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.w"]
    fn __lasx_xvsrlri_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.d"]
    fn __lasx_xvsrlri_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.b"]
    fn __lasx_xvbitclr_b(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.h"]
    fn __lasx_xvbitclr_h(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.w"]
    fn __lasx_xvbitclr_w(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.d"]
    fn __lasx_xvbitclr_d(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.b"]
    fn __lasx_xvbitclri_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.h"]
    fn __lasx_xvbitclri_h(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.w"]
    fn __lasx_xvbitclri_w(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.d"]
    fn __lasx_xvbitclri_d(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitset.b"]
    fn __lasx_xvbitset_b(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitset.h"]
    fn __lasx_xvbitset_h(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitset.w"]
    fn __lasx_xvbitset_w(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitset.d"]
    fn __lasx_xvbitset_d(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.b"]
    fn __lasx_xvbitseti_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.h"]
    fn __lasx_xvbitseti_h(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.w"]
    fn __lasx_xvbitseti_w(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.d"]
    fn __lasx_xvbitseti_d(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.b"]
    fn __lasx_xvbitrev_b(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.h"]
    fn __lasx_xvbitrev_h(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.w"]
    fn __lasx_xvbitrev_w(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.d"]
    fn __lasx_xvbitrev_d(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.b"]
    fn __lasx_xvbitrevi_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.h"]
    fn __lasx_xvbitrevi_h(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.w"]
    fn __lasx_xvbitrevi_w(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.d"]
    fn __lasx_xvbitrevi_d(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvadd.b"]
    fn __lasx_xvadd_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvadd.h"]
    fn __lasx_xvadd_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvadd.w"]
    fn __lasx_xvadd_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvadd.d"]
    fn __lasx_xvadd_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddi.bu"]
    fn __lasx_xvaddi_bu(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvaddi.hu"]
    fn __lasx_xvaddi_hu(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddi.wu"]
    fn __lasx_xvaddi_wu(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddi.du"]
    fn __lasx_xvaddi_du(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsub.b"]
    fn __lasx_xvsub_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsub.h"]
    fn __lasx_xvsub_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsub.w"]
    fn __lasx_xvsub_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsub.d"]
    fn __lasx_xvsub_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubi.bu"]
    fn __lasx_xvsubi_bu(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsubi.hu"]
    fn __lasx_xvsubi_hu(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubi.wu"]
    fn __lasx_xvsubi_wu(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubi.du"]
    fn __lasx_xvsubi_du(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmax.b"]
    fn __lasx_xvmax_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmax.h"]
    fn __lasx_xvmax_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmax.w"]
    fn __lasx_xvmax_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmax.d"]
    fn __lasx_xvmax_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.b"]
    fn __lasx_xvmaxi_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.h"]
    fn __lasx_xvmaxi_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.w"]
    fn __lasx_xvmaxi_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.d"]
    fn __lasx_xvmaxi_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmax.bu"]
    fn __lasx_xvmax_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmax.hu"]
    fn __lasx_xvmax_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmax.wu"]
    fn __lasx_xvmax_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmax.du"]
    fn __lasx_xvmax_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.bu"]
    fn __lasx_xvmaxi_bu(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.hu"]
    fn __lasx_xvmaxi_hu(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.wu"]
    fn __lasx_xvmaxi_wu(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmaxi.du"]
    fn __lasx_xvmaxi_du(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmin.b"]
    fn __lasx_xvmin_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmin.h"]
    fn __lasx_xvmin_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmin.w"]
    fn __lasx_xvmin_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmin.d"]
    fn __lasx_xvmin_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmini.b"]
    fn __lasx_xvmini_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmini.h"]
    fn __lasx_xvmini_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmini.w"]
    fn __lasx_xvmini_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmini.d"]
    fn __lasx_xvmini_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmin.bu"]
    fn __lasx_xvmin_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmin.hu"]
    fn __lasx_xvmin_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmin.wu"]
    fn __lasx_xvmin_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmin.du"]
    fn __lasx_xvmin_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmini.bu"]
    fn __lasx_xvmini_bu(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmini.hu"]
    fn __lasx_xvmini_hu(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmini.wu"]
    fn __lasx_xvmini_wu(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmini.du"]
    fn __lasx_xvmini_du(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvseq.b"]
    fn __lasx_xvseq_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvseq.h"]
    fn __lasx_xvseq_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvseq.w"]
    fn __lasx_xvseq_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvseq.d"]
    fn __lasx_xvseq_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvseqi.b"]
    fn __lasx_xvseqi_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvseqi.h"]
    fn __lasx_xvseqi_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvseqi.w"]
    fn __lasx_xvseqi_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvseqi.d"]
    fn __lasx_xvseqi_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslt.b"]
    fn __lasx_xvslt_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslt.h"]
    fn __lasx_xvslt_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslt.w"]
    fn __lasx_xvslt_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslt.d"]
    fn __lasx_xvslt_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslti.b"]
    fn __lasx_xvslti_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslti.h"]
    fn __lasx_xvslti_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslti.w"]
    fn __lasx_xvslti_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslti.d"]
    fn __lasx_xvslti_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslt.bu"]
    fn __lasx_xvslt_bu(a: v32u8, b: v32u8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslt.hu"]
    fn __lasx_xvslt_hu(a: v16u16, b: v16u16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslt.wu"]
    fn __lasx_xvslt_wu(a: v8u32, b: v8u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslt.du"]
    fn __lasx_xvslt_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslti.bu"]
    fn __lasx_xvslti_bu(a: v32u8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslti.hu"]
    fn __lasx_xvslti_hu(a: v16u16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslti.wu"]
    fn __lasx_xvslti_wu(a: v8u32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslti.du"]
    fn __lasx_xvslti_du(a: v4u64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsle.b"]
    fn __lasx_xvsle_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsle.h"]
    fn __lasx_xvsle_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsle.w"]
    fn __lasx_xvsle_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsle.d"]
    fn __lasx_xvsle_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslei.b"]
    fn __lasx_xvslei_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslei.h"]
    fn __lasx_xvslei_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslei.w"]
    fn __lasx_xvslei_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslei.d"]
    fn __lasx_xvslei_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsle.bu"]
    fn __lasx_xvsle_bu(a: v32u8, b: v32u8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsle.hu"]
    fn __lasx_xvsle_hu(a: v16u16, b: v16u16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsle.wu"]
    fn __lasx_xvsle_wu(a: v8u32, b: v8u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsle.du"]
    fn __lasx_xvsle_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvslei.bu"]
    fn __lasx_xvslei_bu(a: v32u8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvslei.hu"]
    fn __lasx_xvslei_hu(a: v16u16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvslei.wu"]
    fn __lasx_xvslei_wu(a: v8u32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvslei.du"]
    fn __lasx_xvslei_du(a: v4u64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsat.b"]
    fn __lasx_xvsat_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsat.h"]
    fn __lasx_xvsat_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsat.w"]
    fn __lasx_xvsat_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsat.d"]
    fn __lasx_xvsat_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsat.bu"]
    fn __lasx_xvsat_bu(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvsat.hu"]
    fn __lasx_xvsat_hu(a: v16u16, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsat.wu"]
    fn __lasx_xvsat_wu(a: v8u32, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsat.du"]
    fn __lasx_xvsat_du(a: v4u64, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvadda.b"]
    fn __lasx_xvadda_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvadda.h"]
    fn __lasx_xvadda_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvadda.w"]
    fn __lasx_xvadda_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvadda.d"]
    fn __lasx_xvadda_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsadd.b"]
    fn __lasx_xvsadd_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsadd.h"]
    fn __lasx_xvsadd_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsadd.w"]
    fn __lasx_xvsadd_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsadd.d"]
    fn __lasx_xvsadd_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsadd.bu"]
    fn __lasx_xvsadd_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvsadd.hu"]
    fn __lasx_xvsadd_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsadd.wu"]
    fn __lasx_xvsadd_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsadd.du"]
    fn __lasx_xvsadd_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvavg.b"]
    fn __lasx_xvavg_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvavg.h"]
    fn __lasx_xvavg_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvavg.w"]
    fn __lasx_xvavg_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvavg.d"]
    fn __lasx_xvavg_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvavg.bu"]
    fn __lasx_xvavg_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvavg.hu"]
    fn __lasx_xvavg_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvavg.wu"]
    fn __lasx_xvavg_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvavg.du"]
    fn __lasx_xvavg_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvavgr.b"]
    fn __lasx_xvavgr_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvavgr.h"]
    fn __lasx_xvavgr_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvavgr.w"]
    fn __lasx_xvavgr_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvavgr.d"]
    fn __lasx_xvavgr_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvavgr.bu"]
    fn __lasx_xvavgr_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvavgr.hu"]
    fn __lasx_xvavgr_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvavgr.wu"]
    fn __lasx_xvavgr_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvavgr.du"]
    fn __lasx_xvavgr_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssub.b"]
    fn __lasx_xvssub_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssub.h"]
    fn __lasx_xvssub_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssub.w"]
    fn __lasx_xvssub_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssub.d"]
    fn __lasx_xvssub_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssub.bu"]
    fn __lasx_xvssub_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssub.hu"]
    fn __lasx_xvssub_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssub.wu"]
    fn __lasx_xvssub_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssub.du"]
    fn __lasx_xvssub_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvabsd.b"]
    fn __lasx_xvabsd_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvabsd.h"]
    fn __lasx_xvabsd_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvabsd.w"]
    fn __lasx_xvabsd_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvabsd.d"]
    fn __lasx_xvabsd_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvabsd.bu"]
    fn __lasx_xvabsd_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvabsd.hu"]
    fn __lasx_xvabsd_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvabsd.wu"]
    fn __lasx_xvabsd_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvabsd.du"]
    fn __lasx_xvabsd_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmul.b"]
    fn __lasx_xvmul_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmul.h"]
    fn __lasx_xvmul_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmul.w"]
    fn __lasx_xvmul_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmul.d"]
    fn __lasx_xvmul_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmadd.b"]
    fn __lasx_xvmadd_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmadd.h"]
    fn __lasx_xvmadd_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmadd.w"]
    fn __lasx_xvmadd_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmadd.d"]
    fn __lasx_xvmadd_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmsub.b"]
    fn __lasx_xvmsub_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmsub.h"]
    fn __lasx_xvmsub_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmsub.w"]
    fn __lasx_xvmsub_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmsub.d"]
    fn __lasx_xvmsub_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvdiv.b"]
    fn __lasx_xvdiv_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvdiv.h"]
    fn __lasx_xvdiv_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvdiv.w"]
    fn __lasx_xvdiv_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvdiv.d"]
    fn __lasx_xvdiv_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvdiv.bu"]
    fn __lasx_xvdiv_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvdiv.hu"]
    fn __lasx_xvdiv_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvdiv.wu"]
    fn __lasx_xvdiv_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvdiv.du"]
    fn __lasx_xvdiv_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.h.b"]
    fn __lasx_xvhaddw_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.w.h"]
    fn __lasx_xvhaddw_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.d.w"]
    fn __lasx_xvhaddw_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.hu.bu"]
    fn __lasx_xvhaddw_hu_bu(a: v32u8, b: v32u8) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.wu.hu"]
    fn __lasx_xvhaddw_wu_hu(a: v16u16, b: v16u16) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.du.wu"]
    fn __lasx_xvhaddw_du_wu(a: v8u32, b: v8u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.h.b"]
    fn __lasx_xvhsubw_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.w.h"]
    fn __lasx_xvhsubw_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.d.w"]
    fn __lasx_xvhsubw_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.hu.bu"]
    fn __lasx_xvhsubw_hu_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.wu.hu"]
    fn __lasx_xvhsubw_wu_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.du.wu"]
    fn __lasx_xvhsubw_du_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmod.b"]
    fn __lasx_xvmod_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmod.h"]
    fn __lasx_xvmod_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmod.w"]
    fn __lasx_xvmod_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmod.d"]
    fn __lasx_xvmod_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmod.bu"]
    fn __lasx_xvmod_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmod.hu"]
    fn __lasx_xvmod_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmod.wu"]
    fn __lasx_xvmod_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmod.du"]
    fn __lasx_xvmod_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.b"]
    fn __lasx_xvrepl128vei_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.h"]
    fn __lasx_xvrepl128vei_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.w"]
    fn __lasx_xvrepl128vei_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.d"]
    fn __lasx_xvrepl128vei_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickev.b"]
    fn __lasx_xvpickev_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpickev.h"]
    fn __lasx_xvpickev_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpickev.w"]
    fn __lasx_xvpickev_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickev.d"]
    fn __lasx_xvpickev_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickod.b"]
    fn __lasx_xvpickod_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpickod.h"]
    fn __lasx_xvpickod_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpickod.w"]
    fn __lasx_xvpickod_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickod.d"]
    fn __lasx_xvpickod_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvilvh.b"]
    fn __lasx_xvilvh_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvilvh.h"]
    fn __lasx_xvilvh_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvilvh.w"]
    fn __lasx_xvilvh_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvilvh.d"]
    fn __lasx_xvilvh_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvilvl.b"]
    fn __lasx_xvilvl_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvilvl.h"]
    fn __lasx_xvilvl_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvilvl.w"]
    fn __lasx_xvilvl_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvilvl.d"]
    fn __lasx_xvilvl_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpackev.b"]
    fn __lasx_xvpackev_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpackev.h"]
    fn __lasx_xvpackev_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpackev.w"]
    fn __lasx_xvpackev_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpackev.d"]
    fn __lasx_xvpackev_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpackod.b"]
    fn __lasx_xvpackod_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpackod.h"]
    fn __lasx_xvpackod_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpackod.w"]
    fn __lasx_xvpackod_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpackod.d"]
    fn __lasx_xvpackod_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvshuf.b"]
    fn __lasx_xvshuf_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvshuf.h"]
    fn __lasx_xvshuf_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf.w"]
    fn __lasx_xvshuf_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvshuf.d"]
    fn __lasx_xvshuf_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvand.v"]
    fn __lasx_xvand_v(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvandi.b"]
    fn __lasx_xvandi_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvor.v"]
    fn __lasx_xvor_v(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvori.b"]
    fn __lasx_xvori_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvnor.v"]
    fn __lasx_xvnor_v(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvnori.b"]
    fn __lasx_xvnori_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvxor.v"]
    fn __lasx_xvxor_v(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvxori.b"]
    fn __lasx_xvxori_b(a: v32u8, b: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitsel.v"]
    fn __lasx_xvbitsel_v(a: v32u8, b: v32u8, c: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitseli.b"]
    fn __lasx_xvbitseli_b(a: v32u8, b: v32u8, c: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.b"]
    fn __lasx_xvshuf4i_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.h"]
    fn __lasx_xvshuf4i_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.w"]
    fn __lasx_xvshuf4i_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplgr2vr.b"]
    fn __lasx_xvreplgr2vr_b(a: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvreplgr2vr.h"]
    fn __lasx_xvreplgr2vr_h(a: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvreplgr2vr.w"]
    fn __lasx_xvreplgr2vr_w(a: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplgr2vr.d"]
    fn __lasx_xvreplgr2vr_d(a: i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpcnt.b"]
    fn __lasx_xvpcnt_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpcnt.h"]
    fn __lasx_xvpcnt_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpcnt.w"]
    fn __lasx_xvpcnt_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpcnt.d"]
    fn __lasx_xvpcnt_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvclo.b"]
    fn __lasx_xvclo_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvclo.h"]
    fn __lasx_xvclo_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvclo.w"]
    fn __lasx_xvclo_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvclo.d"]
    fn __lasx_xvclo_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvclz.b"]
    fn __lasx_xvclz_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvclz.h"]
    fn __lasx_xvclz_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvclz.w"]
    fn __lasx_xvclz_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvclz.d"]
    fn __lasx_xvclz_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfadd.s"]
    fn __lasx_xvfadd_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfadd.d"]
    fn __lasx_xvfadd_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfsub.s"]
    fn __lasx_xvfsub_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfsub.d"]
    fn __lasx_xvfsub_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmul.s"]
    fn __lasx_xvfmul_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmul.d"]
    fn __lasx_xvfmul_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfdiv.s"]
    fn __lasx_xvfdiv_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfdiv.d"]
    fn __lasx_xvfdiv_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfcvt.h.s"]
    fn __lasx_xvfcvt_h_s(a: v8f32, b: v8f32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvfcvt.s.d"]
    fn __lasx_xvfcvt_s_d(a: v4f64, b: v4f64) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmin.s"]
    fn __lasx_xvfmin_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmin.d"]
    fn __lasx_xvfmin_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmina.s"]
    fn __lasx_xvfmina_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmina.d"]
    fn __lasx_xvfmina_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmax.s"]
    fn __lasx_xvfmax_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmax.d"]
    fn __lasx_xvfmax_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmaxa.s"]
    fn __lasx_xvfmaxa_s(a: v8f32, b: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmaxa.d"]
    fn __lasx_xvfmaxa_d(a: v4f64, b: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfclass.s"]
    fn __lasx_xvfclass_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfclass.d"]
    fn __lasx_xvfclass_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfsqrt.s"]
    fn __lasx_xvfsqrt_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfsqrt.d"]
    fn __lasx_xvfsqrt_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrecip.s"]
    fn __lasx_xvfrecip_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrecip.d"]
    fn __lasx_xvfrecip_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrecipe.s"]
    fn __lasx_xvfrecipe_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrecipe.d"]
    fn __lasx_xvfrecipe_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrte.s"]
    fn __lasx_xvfrsqrte_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrte.d"]
    fn __lasx_xvfrsqrte_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrint.s"]
    fn __lasx_xvfrint_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrint.d"]
    fn __lasx_xvfrint_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrt.s"]
    fn __lasx_xvfrsqrt_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrt.d"]
    fn __lasx_xvfrsqrt_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvflogb.s"]
    fn __lasx_xvflogb_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvflogb.d"]
    fn __lasx_xvflogb_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfcvth.s.h"]
    fn __lasx_xvfcvth_s_h(a: v16i16) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfcvth.d.s"]
    fn __lasx_xvfcvth_d_s(a: v8f32) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfcvtl.s.h"]
    fn __lasx_xvfcvtl_s_h(a: v16i16) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfcvtl.d.s"]
    fn __lasx_xvfcvtl_d_s(a: v8f32) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvftint.w.s"]
    fn __lasx_xvftint_w_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftint.l.d"]
    fn __lasx_xvftint_l_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftint.wu.s"]
    fn __lasx_xvftint_wu_s(a: v8f32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvftint.lu.d"]
    fn __lasx_xvftint_lu_d(a: v4f64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.w.s"]
    fn __lasx_xvftintrz_w_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.l.d"]
    fn __lasx_xvftintrz_l_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.wu.s"]
    fn __lasx_xvftintrz_wu_s(a: v8f32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.lu.d"]
    fn __lasx_xvftintrz_lu_d(a: v4f64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.w"]
    fn __lasx_xvffint_s_w(a: v8i32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvffint.d.l"]
    fn __lasx_xvffint_d_l(a: v4i64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.wu"]
    fn __lasx_xvffint_s_wu(a: v8u32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvffint.d.lu"]
    fn __lasx_xvffint_d_lu(a: v4u64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvreplve.b"]
    fn __lasx_xvreplve_b(a: v32i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvreplve.h"]
    fn __lasx_xvreplve_h(a: v16i16, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvreplve.w"]
    fn __lasx_xvreplve_w(a: v8i32, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplve.d"]
    fn __lasx_xvreplve_d(a: v4i64, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpermi.w"]
    fn __lasx_xvpermi_w(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvandn.v"]
    fn __lasx_xvandn_v(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvneg.b"]
    fn __lasx_xvneg_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvneg.h"]
    fn __lasx_xvneg_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvneg.w"]
    fn __lasx_xvneg_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvneg.d"]
    fn __lasx_xvneg_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmuh.b"]
    fn __lasx_xvmuh_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmuh.h"]
    fn __lasx_xvmuh_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmuh.w"]
    fn __lasx_xvmuh_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmuh.d"]
    fn __lasx_xvmuh_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmuh.bu"]
    fn __lasx_xvmuh_bu(a: v32u8, b: v32u8) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmuh.hu"]
    fn __lasx_xvmuh_hu(a: v16u16, b: v16u16) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmuh.wu"]
    fn __lasx_xvmuh_wu(a: v8u32, b: v8u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmuh.du"]
    fn __lasx_xvmuh_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.h.b"]
    fn __lasx_xvsllwil_h_b(a: v32i8, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.w.h"]
    fn __lasx_xvsllwil_w_h(a: v16i16, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.d.w"]
    fn __lasx_xvsllwil_d_w(a: v8i32, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.hu.bu"]
    fn __lasx_xvsllwil_hu_bu(a: v32u8, b: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.wu.hu"]
    fn __lasx_xvsllwil_wu_hu(a: v16u16, b: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.du.wu"]
    fn __lasx_xvsllwil_du_wu(a: v8u32, b: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsran.b.h"]
    fn __lasx_xvsran_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsran.h.w"]
    fn __lasx_xvsran_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsran.w.d"]
    fn __lasx_xvsran_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssran.b.h"]
    fn __lasx_xvssran_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssran.h.w"]
    fn __lasx_xvssran_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssran.w.d"]
    fn __lasx_xvssran_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssran.bu.h"]
    fn __lasx_xvssran_bu_h(a: v16u16, b: v16u16) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssran.hu.w"]
    fn __lasx_xvssran_hu_w(a: v8u32, b: v8u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssran.wu.d"]
    fn __lasx_xvssran_wu_d(a: v4u64, b: v4u64) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.b.h"]
    fn __lasx_xvsrarn_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.h.w"]
    fn __lasx_xvsrarn_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.w.d"]
    fn __lasx_xvsrarn_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.b.h"]
    fn __lasx_xvssrarn_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.h.w"]
    fn __lasx_xvssrarn_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.w.d"]
    fn __lasx_xvssrarn_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.bu.h"]
    fn __lasx_xvssrarn_bu_h(a: v16u16, b: v16u16) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.hu.w"]
    fn __lasx_xvssrarn_hu_w(a: v8u32, b: v8u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.wu.d"]
    fn __lasx_xvssrarn_wu_d(a: v4u64, b: v4u64) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrln.b.h"]
    fn __lasx_xvsrln_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrln.h.w"]
    fn __lasx_xvsrln_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrln.w.d"]
    fn __lasx_xvsrln_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrln.bu.h"]
    fn __lasx_xvssrln_bu_h(a: v16u16, b: v16u16) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrln.hu.w"]
    fn __lasx_xvssrln_hu_w(a: v8u32, b: v8u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrln.wu.d"]
    fn __lasx_xvssrln_wu_d(a: v4u64, b: v4u64) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.b.h"]
    fn __lasx_xvsrlrn_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.h.w"]
    fn __lasx_xvsrlrn_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.w.d"]
    fn __lasx_xvsrlrn_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.bu.h"]
    fn __lasx_xvssrlrn_bu_h(a: v16u16, b: v16u16) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.hu.w"]
    fn __lasx_xvssrlrn_hu_w(a: v8u32, b: v8u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.wu.d"]
    fn __lasx_xvssrlrn_wu_d(a: v4u64, b: v4u64) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvfrstpi.b"]
    fn __lasx_xvfrstpi_b(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvfrstpi.h"]
    fn __lasx_xvfrstpi_h(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvfrstp.b"]
    fn __lasx_xvfrstp_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvfrstp.h"]
    fn __lasx_xvfrstp_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.d"]
    fn __lasx_xvshuf4i_d(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvbsrl.v"]
    fn __lasx_xvbsrl_v(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvbsll.v"]
    fn __lasx_xvbsll_v(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvextrins.b"]
    fn __lasx_xvextrins_b(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvextrins.h"]
    fn __lasx_xvextrins_h(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvextrins.w"]
    fn __lasx_xvextrins_w(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvextrins.d"]
    fn __lasx_xvextrins_d(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.b"]
    fn __lasx_xvmskltz_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.h"]
    fn __lasx_xvmskltz_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.w"]
    fn __lasx_xvmskltz_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.d"]
    fn __lasx_xvmskltz_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.b"]
    fn __lasx_xvsigncov_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.h"]
    fn __lasx_xvsigncov_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.w"]
    fn __lasx_xvsigncov_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.d"]
    fn __lasx_xvsigncov_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfmadd.s"]
    fn __lasx_xvfmadd_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmadd.d"]
    fn __lasx_xvfmadd_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmsub.s"]
    fn __lasx_xvfmsub_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmsub.d"]
    fn __lasx_xvfmsub_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfnmadd.s"]
    fn __lasx_xvfnmadd_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfnmadd.d"]
    fn __lasx_xvfnmadd_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfnmsub.s"]
    fn __lasx_xvfnmsub_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfnmsub.d"]
    fn __lasx_xvfnmsub_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.w.s"]
    fn __lasx_xvftintrne_w_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.l.d"]
    fn __lasx_xvftintrne_l_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.w.s"]
    fn __lasx_xvftintrp_w_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.l.d"]
    fn __lasx_xvftintrp_l_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.w.s"]
    fn __lasx_xvftintrm_w_s(a: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.l.d"]
    fn __lasx_xvftintrm_l_d(a: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftint.w.d"]
    fn __lasx_xvftint_w_d(a: v4f64, b: v4f64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.l"]
    fn __lasx_xvffint_s_l(a: v4i64, b: v4i64) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.w.d"]
    fn __lasx_xvftintrz_w_d(a: v4f64, b: v4f64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.w.d"]
    fn __lasx_xvftintrp_w_d(a: v4f64, b: v4f64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.w.d"]
    fn __lasx_xvftintrm_w_d(a: v4f64, b: v4f64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.w.d"]
    fn __lasx_xvftintrne_w_d(a: v4f64, b: v4f64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftinth.l.s"]
    fn __lasx_xvftinth_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintl.l.s"]
    fn __lasx_xvftintl_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvffinth.d.w"]
    fn __lasx_xvffinth_d_w(a: v8i32) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvffintl.d.w"]
    fn __lasx_xvffintl_d_w(a: v8i32) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvftintrzh.l.s"]
    fn __lasx_xvftintrzh_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrzl.l.s"]
    fn __lasx_xvftintrzl_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrph.l.s"]
    fn __lasx_xvftintrph_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrpl.l.s"]
    fn __lasx_xvftintrpl_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrmh.l.s"]
    fn __lasx_xvftintrmh_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrml.l.s"]
    fn __lasx_xvftintrml_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrneh.l.s"]
    fn __lasx_xvftintrneh_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrnel.l.s"]
    fn __lasx_xvftintrnel_l_s(a: v8f32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrne.s"]
    fn __lasx_xvfrintrne_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrne.d"]
    fn __lasx_xvfrintrne_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrz.s"]
    fn __lasx_xvfrintrz_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrz.d"]
    fn __lasx_xvfrintrz_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrp.s"]
    fn __lasx_xvfrintrp_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrp.d"]
    fn __lasx_xvfrintrp_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrm.s"]
    fn __lasx_xvfrintrm_s(a: v8f32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrm.d"]
    fn __lasx_xvfrintrm_d(a: v4f64) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvld"]
    fn __lasx_xvld(a: *const i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvst"]
    fn __lasx_xvst(a: v32i8, b: *mut i8, c: i32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.b"]
    fn __lasx_xvstelm_b(a: v32i8, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.h"]
    fn __lasx_xvstelm_h(a: v16i16, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.w"]
    fn __lasx_xvstelm_w(a: v8i32, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.d"]
    fn __lasx_xvstelm_d(a: v4i64, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvinsve0.w"]
    fn __lasx_xvinsve0_w(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvinsve0.d"]
    fn __lasx_xvinsve0_d(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickve.w"]
    fn __lasx_xvpickve_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickve.d"]
    fn __lasx_xvpickve_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.b.h"]
    fn __lasx_xvssrlrn_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.h.w"]
    fn __lasx_xvssrlrn_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.w.d"]
    fn __lasx_xvssrlrn_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrln.b.h"]
    fn __lasx_xvssrln_b_h(a: v16i16, b: v16i16) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrln.h.w"]
    fn __lasx_xvssrln_h_w(a: v8i32, b: v8i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrln.w.d"]
    fn __lasx_xvssrln_w_d(a: v4i64, b: v4i64) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvorn.v"]
    fn __lasx_xvorn_v(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvldi"]
    fn __lasx_xvldi(a: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvldx"]
    fn __lasx_xvldx(a: *const i8, b: i64) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvstx"]
    fn __lasx_xvstx(a: v32i8, b: *mut i8, c: i64);
    #[link_name = "llvm.loongarch.lasx.xvextl.qu.du"]
    fn __lasx_xvextl_qu_du(a: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvinsgr2vr.w"]
    fn __lasx_xvinsgr2vr_w(a: v8i32, b: i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvinsgr2vr.d"]
    fn __lasx_xvinsgr2vr_d(a: v4i64, b: i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.b"]
    fn __lasx_xvreplve0_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.h"]
    fn __lasx_xvreplve0_h(a: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.w"]
    fn __lasx_xvreplve0_w(a: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.d"]
    fn __lasx_xvreplve0_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.q"]
    fn __lasx_xvreplve0_q(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.vext2xv.h.b"]
    fn __lasx_vext2xv_h_b(a: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.vext2xv.w.h"]
    fn __lasx_vext2xv_w_h(a: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.w"]
    fn __lasx_vext2xv_d_w(a: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.w.b"]
    fn __lasx_vext2xv_w_b(a: v32i8) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.h"]
    fn __lasx_vext2xv_d_h(a: v16i16) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.b"]
    fn __lasx_vext2xv_d_b(a: v32i8) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.hu.bu"]
    fn __lasx_vext2xv_hu_bu(a: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.vext2xv.wu.hu"]
    fn __lasx_vext2xv_wu_hu(a: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.wu"]
    fn __lasx_vext2xv_du_wu(a: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.wu.bu"]
    fn __lasx_vext2xv_wu_bu(a: v32i8) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.hu"]
    fn __lasx_vext2xv_du_hu(a: v16i16) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.bu"]
    fn __lasx_vext2xv_du_bu(a: v32i8) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpermi.q"]
    fn __lasx_xvpermi_q(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpermi.d"]
    fn __lasx_xvpermi_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvperm.w"]
    fn __lasx_xvperm_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.b"]
    fn __lasx_xvldrepl_b(a: *const i8, b: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.h"]
    fn __lasx_xvldrepl_h(a: *const i8, b: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.w"]
    fn __lasx_xvldrepl_w(a: *const i8, b: i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.d"]
    fn __lasx_xvldrepl_d(a: *const i8, b: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickve2gr.w"]
    fn __lasx_xvpickve2gr_w(a: v8i32, b: u32) -> i32;
    #[link_name = "llvm.loongarch.lasx.xvpickve2gr.wu"]
    fn __lasx_xvpickve2gr_wu(a: v8i32, b: u32) -> u32;
    #[link_name = "llvm.loongarch.lasx.xvpickve2gr.d"]
    fn __lasx_xvpickve2gr_d(a: v4i64, b: u32) -> i64;
    #[link_name = "llvm.loongarch.lasx.xvpickve2gr.du"]
    fn __lasx_xvpickve2gr_du(a: v4i64, b: u32) -> u64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.d"]
    fn __lasx_xvaddwev_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.w"]
    fn __lasx_xvaddwev_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.h"]
    fn __lasx_xvaddwev_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.b"]
    fn __lasx_xvaddwev_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.du"]
    fn __lasx_xvaddwev_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.wu"]
    fn __lasx_xvaddwev_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.hu"]
    fn __lasx_xvaddwev_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.bu"]
    fn __lasx_xvaddwev_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.q.d"]
    fn __lasx_xvsubwev_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.d.w"]
    fn __lasx_xvsubwev_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.w.h"]
    fn __lasx_xvsubwev_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.h.b"]
    fn __lasx_xvsubwev_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.q.du"]
    fn __lasx_xvsubwev_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.d.wu"]
    fn __lasx_xvsubwev_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.w.hu"]
    fn __lasx_xvsubwev_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.h.bu"]
    fn __lasx_xvsubwev_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.d"]
    fn __lasx_xvmulwev_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.w"]
    fn __lasx_xvmulwev_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.h"]
    fn __lasx_xvmulwev_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.b"]
    fn __lasx_xvmulwev_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.du"]
    fn __lasx_xvmulwev_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.wu"]
    fn __lasx_xvmulwev_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.hu"]
    fn __lasx_xvmulwev_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.bu"]
    fn __lasx_xvmulwev_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.d"]
    fn __lasx_xvaddwod_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.w"]
    fn __lasx_xvaddwod_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.h"]
    fn __lasx_xvaddwod_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.b"]
    fn __lasx_xvaddwod_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.du"]
    fn __lasx_xvaddwod_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.wu"]
    fn __lasx_xvaddwod_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.hu"]
    fn __lasx_xvaddwod_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.bu"]
    fn __lasx_xvaddwod_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.q.d"]
    fn __lasx_xvsubwod_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.d.w"]
    fn __lasx_xvsubwod_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.w.h"]
    fn __lasx_xvsubwod_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.h.b"]
    fn __lasx_xvsubwod_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.q.du"]
    fn __lasx_xvsubwod_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.d.wu"]
    fn __lasx_xvsubwod_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.w.hu"]
    fn __lasx_xvsubwod_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.h.bu"]
    fn __lasx_xvsubwod_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.d"]
    fn __lasx_xvmulwod_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.w"]
    fn __lasx_xvmulwod_d_w(a: v8i32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.h"]
    fn __lasx_xvmulwod_w_h(a: v16i16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.b"]
    fn __lasx_xvmulwod_h_b(a: v32i8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.du"]
    fn __lasx_xvmulwod_q_du(a: v4u64, b: v4u64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.wu"]
    fn __lasx_xvmulwod_d_wu(a: v8u32, b: v8u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.hu"]
    fn __lasx_xvmulwod_w_hu(a: v16u16, b: v16u16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.bu"]
    fn __lasx_xvmulwod_h_bu(a: v32u8, b: v32u8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.wu.w"]
    fn __lasx_xvaddwev_d_wu_w(a: v8u32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.hu.h"]
    fn __lasx_xvaddwev_w_hu_h(a: v16u16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.bu.b"]
    fn __lasx_xvaddwev_h_bu_b(a: v32u8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.wu.w"]
    fn __lasx_xvmulwev_d_wu_w(a: v8u32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.hu.h"]
    fn __lasx_xvmulwev_w_hu_h(a: v16u16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.bu.b"]
    fn __lasx_xvmulwev_h_bu_b(a: v32u8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.wu.w"]
    fn __lasx_xvaddwod_d_wu_w(a: v8u32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.hu.h"]
    fn __lasx_xvaddwod_w_hu_h(a: v16u16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.bu.b"]
    fn __lasx_xvaddwod_h_bu_b(a: v32u8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.wu.w"]
    fn __lasx_xvmulwod_d_wu_w(a: v8u32, b: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.hu.h"]
    fn __lasx_xvmulwod_w_hu_h(a: v16u16, b: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.bu.b"]
    fn __lasx_xvmulwod_h_bu_b(a: v32u8, b: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.q.d"]
    fn __lasx_xvhaddw_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.qu.du"]
    fn __lasx_xvhaddw_qu_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.q.d"]
    fn __lasx_xvhsubw_q_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.qu.du"]
    fn __lasx_xvhsubw_qu_du(a: v4u64, b: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.d"]
    fn __lasx_xvmaddwev_q_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.w"]
    fn __lasx_xvmaddwev_d_w(a: v4i64, b: v8i32, c: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.h"]
    fn __lasx_xvmaddwev_w_h(a: v8i32, b: v16i16, c: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.b"]
    fn __lasx_xvmaddwev_h_b(a: v16i16, b: v32i8, c: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.du"]
    fn __lasx_xvmaddwev_q_du(a: v4u64, b: v4u64, c: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.wu"]
    fn __lasx_xvmaddwev_d_wu(a: v4u64, b: v8u32, c: v8u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.hu"]
    fn __lasx_xvmaddwev_w_hu(a: v8u32, b: v16u16, c: v16u16) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.bu"]
    fn __lasx_xvmaddwev_h_bu(a: v16u16, b: v32u8, c: v32u8) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.d"]
    fn __lasx_xvmaddwod_q_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.w"]
    fn __lasx_xvmaddwod_d_w(a: v4i64, b: v8i32, c: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.h"]
    fn __lasx_xvmaddwod_w_h(a: v8i32, b: v16i16, c: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.b"]
    fn __lasx_xvmaddwod_h_b(a: v16i16, b: v32i8, c: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.du"]
    fn __lasx_xvmaddwod_q_du(a: v4u64, b: v4u64, c: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.wu"]
    fn __lasx_xvmaddwod_d_wu(a: v4u64, b: v8u32, c: v8u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.hu"]
    fn __lasx_xvmaddwod_w_hu(a: v8u32, b: v16u16, c: v16u16) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.bu"]
    fn __lasx_xvmaddwod_h_bu(a: v16u16, b: v32u8, c: v32u8) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.du.d"]
    fn __lasx_xvmaddwev_q_du_d(a: v4i64, b: v4u64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.wu.w"]
    fn __lasx_xvmaddwev_d_wu_w(a: v4i64, b: v8u32, c: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.hu.h"]
    fn __lasx_xvmaddwev_w_hu_h(a: v8i32, b: v16u16, c: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.bu.b"]
    fn __lasx_xvmaddwev_h_bu_b(a: v16i16, b: v32u8, c: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.du.d"]
    fn __lasx_xvmaddwod_q_du_d(a: v4i64, b: v4u64, c: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.wu.w"]
    fn __lasx_xvmaddwod_d_wu_w(a: v4i64, b: v8u32, c: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.hu.h"]
    fn __lasx_xvmaddwod_w_hu_h(a: v8i32, b: v16u16, c: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.bu.b"]
    fn __lasx_xvmaddwod_h_bu_b(a: v16i16, b: v32u8, c: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotr.b"]
    fn __lasx_xvrotr_b(a: v32i8, b: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrotr.h"]
    fn __lasx_xvrotr_h(a: v16i16, b: v16i16) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotr.w"]
    fn __lasx_xvrotr_w(a: v8i32, b: v8i32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrotr.d"]
    fn __lasx_xvrotr_d(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvadd.q"]
    fn __lasx_xvadd_q(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsub.q"]
    fn __lasx_xvsub_q(a: v4i64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.du.d"]
    fn __lasx_xvaddwev_q_du_d(a: v4u64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.du.d"]
    fn __lasx_xvaddwod_q_du_d(a: v4u64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.du.d"]
    fn __lasx_xvmulwev_q_du_d(a: v4u64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.du.d"]
    fn __lasx_xvmulwod_q_du_d(a: v4u64, b: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmskgez.b"]
    fn __lasx_xvmskgez_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmsknz.b"]
    fn __lasx_xvmsknz_b(a: v32i8) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvexth.h.b"]
    fn __lasx_xvexth_h_b(a: v32i8) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvexth.w.h"]
    fn __lasx_xvexth_w_h(a: v16i16) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvexth.d.w"]
    fn __lasx_xvexth_d_w(a: v8i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvexth.q.d"]
    fn __lasx_xvexth_q_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvexth.hu.bu"]
    fn __lasx_xvexth_hu_bu(a: v32u8) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvexth.wu.hu"]
    fn __lasx_xvexth_wu_hu(a: v16u16) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvexth.du.wu"]
    fn __lasx_xvexth_du_wu(a: v8u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvexth.qu.du"]
    fn __lasx_xvexth_qu_du(a: v4u64) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvrotri.b"]
    fn __lasx_xvrotri_b(a: v32i8, b: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrotri.h"]
    fn __lasx_xvrotri_h(a: v16i16, b: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotri.w"]
    fn __lasx_xvrotri_w(a: v8i32, b: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrotri.d"]
    fn __lasx_xvrotri_d(a: v4i64, b: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvextl.q.d"]
    fn __lasx_xvextl_q_d(a: v4i64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.b.h"]
    fn __lasx_xvsrlni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.h.w"]
    fn __lasx_xvsrlni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.w.d"]
    fn __lasx_xvsrlni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.d.q"]
    fn __lasx_xvsrlni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.b.h"]
    fn __lasx_xvsrlrni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.h.w"]
    fn __lasx_xvsrlrni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.w.d"]
    fn __lasx_xvsrlrni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.d.q"]
    fn __lasx_xvsrlrni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.b.h"]
    fn __lasx_xvssrlni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.h.w"]
    fn __lasx_xvssrlni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.w.d"]
    fn __lasx_xvssrlni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.d.q"]
    fn __lasx_xvssrlni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.bu.h"]
    fn __lasx_xvssrlni_bu_h(a: v32u8, b: v32i8, c: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.hu.w"]
    fn __lasx_xvssrlni_hu_w(a: v16u16, b: v16i16, c: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.wu.d"]
    fn __lasx_xvssrlni_wu_d(a: v8u32, b: v8i32, c: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.du.q"]
    fn __lasx_xvssrlni_du_q(a: v4u64, b: v4i64, c: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.b.h"]
    fn __lasx_xvssrlrni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.h.w"]
    fn __lasx_xvssrlrni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.w.d"]
    fn __lasx_xvssrlrni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.d.q"]
    fn __lasx_xvssrlrni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.bu.h"]
    fn __lasx_xvssrlrni_bu_h(a: v32u8, b: v32i8, c: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.hu.w"]
    fn __lasx_xvssrlrni_hu_w(a: v16u16, b: v16i16, c: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.wu.d"]
    fn __lasx_xvssrlrni_wu_d(a: v8u32, b: v8i32, c: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.du.q"]
    fn __lasx_xvssrlrni_du_q(a: v4u64, b: v4i64, c: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsrani.b.h"]
    fn __lasx_xvsrani_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrani.h.w"]
    fn __lasx_xvsrani_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrani.w.d"]
    fn __lasx_xvsrani_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrani.d.q"]
    fn __lasx_xvsrani_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.b.h"]
    fn __lasx_xvsrarni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.h.w"]
    fn __lasx_xvsrarni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.w.d"]
    fn __lasx_xvsrarni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.d.q"]
    fn __lasx_xvsrarni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrani.b.h"]
    fn __lasx_xvssrani_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrani.h.w"]
    fn __lasx_xvssrani_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrani.w.d"]
    fn __lasx_xvssrani_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrani.d.q"]
    fn __lasx_xvssrani_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrani.bu.h"]
    fn __lasx_xvssrani_bu_h(a: v32u8, b: v32i8, c: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrani.hu.w"]
    fn __lasx_xvssrani_hu_w(a: v16u16, b: v16i16, c: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrani.wu.d"]
    fn __lasx_xvssrani_wu_d(a: v8u32, b: v8i32, c: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrani.du.q"]
    fn __lasx_xvssrani_du_q(a: v4u64, b: v4i64, c: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.b.h"]
    fn __lasx_xvssrarni_b_h(a: v32i8, b: v32i8, c: u32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.h.w"]
    fn __lasx_xvssrarni_h_w(a: v16i16, b: v16i16, c: u32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.w.d"]
    fn __lasx_xvssrarni_w_d(a: v8i32, b: v8i32, c: u32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.d.q"]
    fn __lasx_xvssrarni_d_q(a: v4i64, b: v4i64, c: u32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.bu.h"]
    fn __lasx_xvssrarni_bu_h(a: v32u8, b: v32i8, c: u32) -> v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.hu.w"]
    fn __lasx_xvssrarni_hu_w(a: v16u16, b: v16i16, c: u32) -> v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.wu.d"]
    fn __lasx_xvssrarni_wu_d(a: v8u32, b: v8i32, c: u32) -> v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.du.q"]
    fn __lasx_xvssrarni_du_q(a: v4u64, b: v4i64, c: u32) -> v4u64;
    #[link_name = "llvm.loongarch.lasx.xbnz.b"]
    fn __lasx_xbnz_b(a: v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.d"]
    fn __lasx_xbnz_d(a: v4u64) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.h"]
    fn __lasx_xbnz_h(a: v16u16) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.v"]
    fn __lasx_xbnz_v(a: v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.w"]
    fn __lasx_xbnz_w(a: v8u32) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.b"]
    fn __lasx_xbz_b(a: v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.d"]
    fn __lasx_xbz_d(a: v4u64) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.h"]
    fn __lasx_xbz_h(a: v16u16) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.v"]
    fn __lasx_xbz_v(a: v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.w"]
    fn __lasx_xbz_w(a: v8u32) -> i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.caf.d"]
    fn __lasx_xvfcmp_caf_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.caf.s"]
    fn __lasx_xvfcmp_caf_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.ceq.d"]
    fn __lasx_xvfcmp_ceq_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.ceq.s"]
    fn __lasx_xvfcmp_ceq_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cle.d"]
    fn __lasx_xvfcmp_cle_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cle.s"]
    fn __lasx_xvfcmp_cle_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.clt.d"]
    fn __lasx_xvfcmp_clt_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.clt.s"]
    fn __lasx_xvfcmp_clt_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cne.d"]
    fn __lasx_xvfcmp_cne_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cne.s"]
    fn __lasx_xvfcmp_cne_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cor.d"]
    fn __lasx_xvfcmp_cor_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cor.s"]
    fn __lasx_xvfcmp_cor_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cueq.d"]
    fn __lasx_xvfcmp_cueq_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cueq.s"]
    fn __lasx_xvfcmp_cueq_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cule.d"]
    fn __lasx_xvfcmp_cule_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cule.s"]
    fn __lasx_xvfcmp_cule_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cult.d"]
    fn __lasx_xvfcmp_cult_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cult.s"]
    fn __lasx_xvfcmp_cult_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cun.d"]
    fn __lasx_xvfcmp_cun_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cune.d"]
    fn __lasx_xvfcmp_cune_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cune.s"]
    fn __lasx_xvfcmp_cune_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cun.s"]
    fn __lasx_xvfcmp_cun_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.saf.d"]
    fn __lasx_xvfcmp_saf_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.saf.s"]
    fn __lasx_xvfcmp_saf_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.seq.d"]
    fn __lasx_xvfcmp_seq_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.seq.s"]
    fn __lasx_xvfcmp_seq_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sle.d"]
    fn __lasx_xvfcmp_sle_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sle.s"]
    fn __lasx_xvfcmp_sle_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.slt.d"]
    fn __lasx_xvfcmp_slt_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.slt.s"]
    fn __lasx_xvfcmp_slt_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sne.d"]
    fn __lasx_xvfcmp_sne_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sne.s"]
    fn __lasx_xvfcmp_sne_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sor.d"]
    fn __lasx_xvfcmp_sor_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sor.s"]
    fn __lasx_xvfcmp_sor_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sueq.d"]
    fn __lasx_xvfcmp_sueq_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sueq.s"]
    fn __lasx_xvfcmp_sueq_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sule.d"]
    fn __lasx_xvfcmp_sule_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sule.s"]
    fn __lasx_xvfcmp_sule_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sult.d"]
    fn __lasx_xvfcmp_sult_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sult.s"]
    fn __lasx_xvfcmp_sult_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sun.d"]
    fn __lasx_xvfcmp_sun_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sune.d"]
    fn __lasx_xvfcmp_sune_d(a: v4f64, b: v4f64) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sune.s"]
    fn __lasx_xvfcmp_sune_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sun.s"]
    fn __lasx_xvfcmp_sun_s(a: v8f32, b: v8f32) -> v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickve.d.f"]
    fn __lasx_xvpickve_d_f(a: v4f64, b: u32) -> v4f64;
    #[link_name = "llvm.loongarch.lasx.xvpickve.w.f"]
    fn __lasx_xvpickve_w_f(a: v8f32, b: u32) -> v8f32;
    #[link_name = "llvm.loongarch.lasx.xvrepli.b"]
    fn __lasx_xvrepli_b(a: i32) -> v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrepli.d"]
    fn __lasx_xvrepli_d(a: i32) -> v4i64;
    #[link_name = "llvm.loongarch.lasx.xvrepli.h"]
    fn __lasx_xvrepli_h(a: i32) -> v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrepli.w"]
    fn __lasx_xvrepli_w(a: i32) -> v8i32;
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsll_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsll_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsll_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsll_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsll_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsll_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsll_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsll_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslli_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvslli_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslli_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvslli_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslli_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslli_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslli_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvslli_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsra_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsra_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsra_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsra_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsra_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsra_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsra_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsra_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrai_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsrai_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrai_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrai_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrai_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrai_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrai_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrai_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrar_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsrar_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrar_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsrar_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrar_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsrar_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrar_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsrar_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrari_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsrari_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrari_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrari_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrari_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrari_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrari_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrari_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrl_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsrl_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrl_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsrl_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrl_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsrl_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrl_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsrl_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrli_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsrli_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrli_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrli_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrli_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrli_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrli_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrli_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlr_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsrlr_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlr_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsrlr_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlr_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsrlr_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlr_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsrlr_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlri_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsrlri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlri_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrlri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlri_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrlri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlri_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrlri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclr_b(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvbitclr_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclr_h(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvbitclr_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclr_w(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvbitclr_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclr_d(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvbitclr_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclri_b<const IMM3: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvbitclri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclri_h<const IMM4: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvbitclri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclri_w<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvbitclri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitclri_d<const IMM6: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvbitclri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitset_b(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvbitset_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitset_h(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvbitset_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitset_w(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvbitset_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitset_d(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvbitset_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitseti_b<const IMM3: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvbitseti_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitseti_h<const IMM4: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvbitseti_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitseti_w<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvbitseti_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitseti_d<const IMM6: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvbitseti_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrev_b(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvbitrev_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrev_h(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvbitrev_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrev_w(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvbitrev_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrev_d(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvbitrev_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrevi_b<const IMM3: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvbitrevi_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrevi_h<const IMM4: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvbitrevi_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrevi_w<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvbitrevi_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitrevi_d<const IMM6: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvbitrevi_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadd_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvadd_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadd_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvadd_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadd_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvadd_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadd_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddi_bu<const IMM5: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvaddi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddi_hu<const IMM5: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvaddi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddi_wu<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvaddi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddi_du<const IMM5: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvaddi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsub_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsub_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsub_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsub_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsub_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsub_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsub_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsub_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubi_bu<const IMM5: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsubi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubi_hu<const IMM5: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsubi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubi_wu<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsubi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubi_du<const IMM5: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsubi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvmax_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvmax_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvmax_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmax_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_b<const IMM_S5: i32>(a: v32i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmaxi_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_h<const IMM_S5: i32>(a: v16i16) -> v16i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmaxi_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_w<const IMM_S5: i32>(a: v8i32) -> v8i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmaxi_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_d<const IMM_S5: i32>(a: v4i64) -> v4i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmaxi_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvmax_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvmax_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvmax_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmax_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvmax_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_bu<const IMM5: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmaxi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_hu<const IMM5: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmaxi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_wu<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmaxi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaxi_du<const IMM5: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmaxi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvmin_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvmin_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvmin_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmin_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_b<const IMM_S5: i32>(a: v32i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmini_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_h<const IMM_S5: i32>(a: v16i16) -> v16i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmini_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_w<const IMM_S5: i32>(a: v8i32) -> v8i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmini_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_d<const IMM_S5: i32>(a: v4i64) -> v4i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvmini_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvmin_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvmin_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvmin_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmin_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvmin_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_bu<const IMM5: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmini_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_hu<const IMM5: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmini_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_wu<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmini_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmini_du<const IMM5: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvmini_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseq_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvseq_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseq_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvseq_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseq_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvseq_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseq_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvseq_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseqi_b<const IMM_S5: i32>(a: v32i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvseqi_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseqi_h<const IMM_S5: i32>(a: v16i16) -> v16i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvseqi_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseqi_w<const IMM_S5: i32>(a: v8i32) -> v8i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvseqi_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvseqi_d<const IMM_S5: i32>(a: v4i64) -> v4i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvseqi_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvslt_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvslt_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvslt_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvslt_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_b<const IMM_S5: i32>(a: v32i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslti_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_h<const IMM_S5: i32>(a: v16i16) -> v16i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslti_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_w<const IMM_S5: i32>(a: v8i32) -> v8i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslti_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_d<const IMM_S5: i32>(a: v4i64) -> v4i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslti_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_bu(a: v32u8, b: v32u8) -> v32i8 {
    __lasx_xvslt_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_hu(a: v16u16, b: v16u16) -> v16i16 {
    __lasx_xvslt_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_wu(a: v8u32, b: v8u32) -> v8i32 {
    __lasx_xvslt_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslt_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvslt_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_bu<const IMM5: u32>(a: v32u8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslti_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_hu<const IMM5: u32>(a: v16u16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslti_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_wu<const IMM5: u32>(a: v8u32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslti_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslti_du<const IMM5: u32>(a: v4u64) -> v4i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslti_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsle_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsle_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsle_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsle_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_b<const IMM_S5: i32>(a: v32i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslei_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_h<const IMM_S5: i32>(a: v16i16) -> v16i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslei_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_w<const IMM_S5: i32>(a: v8i32) -> v8i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslei_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_d<const IMM_S5: i32>(a: v4i64) -> v4i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lasx_xvslei_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_bu(a: v32u8, b: v32u8) -> v32i8 {
    __lasx_xvsle_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_hu(a: v16u16, b: v16u16) -> v16i16 {
    __lasx_xvsle_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_wu(a: v8u32, b: v8u32) -> v8i32 {
    __lasx_xvsle_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsle_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvsle_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_bu<const IMM5: u32>(a: v32u8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslei_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_hu<const IMM5: u32>(a: v16u16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslei_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_wu<const IMM5: u32>(a: v8u32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslei_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvslei_du<const IMM5: u32>(a: v4u64) -> v4i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvslei_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsat_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsat_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsat_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsat_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_bu<const IMM3: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsat_bu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_hu<const IMM4: u32>(a: v16u16) -> v16u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsat_hu(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_wu<const IMM5: u32>(a: v8u32) -> v8u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsat_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsat_du<const IMM6: u32>(a: v4u64) -> v4u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsat_du(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadda_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvadda_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadda_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvadda_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadda_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvadda_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadda_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvadda_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsadd_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsadd_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsadd_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvsadd_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvsadd_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvsadd_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsadd_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvsadd_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvavg_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvavg_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvavg_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvavg_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvavg_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvavg_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvavg_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavg_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvavg_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvavgr_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvavgr_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvavgr_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvavgr_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvavgr_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvavgr_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvavgr_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvavgr_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvavgr_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvssub_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvssub_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvssub_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvssub_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvssub_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvssub_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvssub_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssub_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvssub_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvabsd_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvabsd_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvabsd_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvabsd_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvabsd_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvabsd_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvabsd_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvabsd_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvabsd_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmul_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvmul_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmul_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvmul_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmul_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvmul_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmul_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmul_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmadd_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8 {
    __lasx_xvmadd_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmadd_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16 {
    __lasx_xvmadd_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmadd_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32 {
    __lasx_xvmadd_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmadd_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    __lasx_xvmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmsub_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8 {
    __lasx_xvmsub_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmsub_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16 {
    __lasx_xvmsub_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmsub_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32 {
    __lasx_xvmsub_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmsub_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    __lasx_xvmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvdiv_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvdiv_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvdiv_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvdiv_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvdiv_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvdiv_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvdiv_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvdiv_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvdiv_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvhaddw_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvhaddw_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvhaddw_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_hu_bu(a: v32u8, b: v32u8) -> v16u16 {
    __lasx_xvhaddw_hu_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_wu_hu(a: v16u16, b: v16u16) -> v8u32 {
    __lasx_xvhaddw_wu_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_du_wu(a: v8u32, b: v8u32) -> v4u64 {
    __lasx_xvhaddw_du_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvhsubw_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvhsubw_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvhsubw_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_hu_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvhsubw_hu_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_wu_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvhsubw_wu_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_du_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvhsubw_du_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvmod_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvmod_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvmod_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmod_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvmod_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvmod_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvmod_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmod_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvmod_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepl128vei_b<const IMM4: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvrepl128vei_b(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepl128vei_h<const IMM3: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvrepl128vei_h(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepl128vei_w<const IMM2: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvrepl128vei_w(a, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepl128vei_d<const IMM1: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM1, 1);
    __lasx_xvrepl128vei_d(a, IMM1)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickev_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvpickev_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickev_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvpickev_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickev_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvpickev_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickev_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvpickev_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickod_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvpickod_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickod_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvpickod_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickod_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvpickod_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickod_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvpickod_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvh_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvilvh_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvh_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvilvh_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvh_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvilvh_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvh_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvilvh_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvl_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvilvl_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvl_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvilvl_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvl_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvilvl_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvilvl_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvilvl_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackev_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvpackev_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackev_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvpackev_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackev_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvpackev_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackev_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvpackev_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackod_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvpackod_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackod_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvpackod_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackod_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvpackod_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpackod_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvpackod_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8 {
    __lasx_xvshuf_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16 {
    __lasx_xvshuf_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf_w(a: v8i32, b: v8i32, c: v8i32) -> v8i32 {
    __lasx_xvshuf_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    __lasx_xvshuf_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvand_v(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvand_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvandi_b<const IMM8: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvandi_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvor_v(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvor_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvori_b<const IMM8: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvnor_v(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvnor_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvnori_b<const IMM8: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvnori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvxor_v(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvxor_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvxori_b<const IMM8: u32>(a: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvxori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitsel_v(a: v32u8, b: v32u8, c: v32u8) -> v32u8 {
    __lasx_xvbitsel_v(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbitseli_b<const IMM8: u32>(a: v32u8, b: v32u8) -> v32u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvbitseli_b(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf4i_b<const IMM8: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvshuf4i_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf4i_h<const IMM8: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvshuf4i_h(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf4i_w<const IMM8: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvshuf4i_w(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplgr2vr_b(a: i32) -> v32i8 {
    __lasx_xvreplgr2vr_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplgr2vr_h(a: i32) -> v16i16 {
    __lasx_xvreplgr2vr_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplgr2vr_w(a: i32) -> v8i32 {
    __lasx_xvreplgr2vr_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplgr2vr_d(a: i64) -> v4i64 {
    __lasx_xvreplgr2vr_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpcnt_b(a: v32i8) -> v32i8 {
    __lasx_xvpcnt_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpcnt_h(a: v16i16) -> v16i16 {
    __lasx_xvpcnt_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpcnt_w(a: v8i32) -> v8i32 {
    __lasx_xvpcnt_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpcnt_d(a: v4i64) -> v4i64 {
    __lasx_xvpcnt_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclo_b(a: v32i8) -> v32i8 {
    __lasx_xvclo_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclo_h(a: v16i16) -> v16i16 {
    __lasx_xvclo_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclo_w(a: v8i32) -> v8i32 {
    __lasx_xvclo_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclo_d(a: v4i64) -> v4i64 {
    __lasx_xvclo_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclz_b(a: v32i8) -> v32i8 {
    __lasx_xvclz_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclz_h(a: v16i16) -> v16i16 {
    __lasx_xvclz_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclz_w(a: v8i32) -> v8i32 {
    __lasx_xvclz_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvclz_d(a: v4i64) -> v4i64 {
    __lasx_xvclz_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfadd_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfadd_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfadd_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfsub_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfsub_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfsub_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfsub_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmul_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfmul_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmul_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfmul_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfdiv_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfdiv_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfdiv_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfdiv_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvt_h_s(a: v8f32, b: v8f32) -> v16i16 {
    __lasx_xvfcvt_h_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvt_s_d(a: v4f64, b: v4f64) -> v8f32 {
    __lasx_xvfcvt_s_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmin_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfmin_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmin_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfmin_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmina_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfmina_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmina_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfmina_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmax_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfmax_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmax_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfmax_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmaxa_s(a: v8f32, b: v8f32) -> v8f32 {
    __lasx_xvfmaxa_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmaxa_d(a: v4f64, b: v4f64) -> v4f64 {
    __lasx_xvfmaxa_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfclass_s(a: v8f32) -> v8i32 {
    __lasx_xvfclass_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfclass_d(a: v4f64) -> v4i64 {
    __lasx_xvfclass_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfsqrt_s(a: v8f32) -> v8f32 {
    __lasx_xvfsqrt_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfsqrt_d(a: v4f64) -> v4f64 {
    __lasx_xvfsqrt_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrecip_s(a: v8f32) -> v8f32 {
    __lasx_xvfrecip_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrecip_d(a: v4f64) -> v4f64 {
    __lasx_xvfrecip_d(a)
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrecipe_s(a: v8f32) -> v8f32 {
    __lasx_xvfrecipe_s(a)
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrecipe_d(a: v4f64) -> v4f64 {
    __lasx_xvfrecipe_d(a)
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrsqrte_s(a: v8f32) -> v8f32 {
    __lasx_xvfrsqrte_s(a)
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrsqrte_d(a: v4f64) -> v4f64 {
    __lasx_xvfrsqrte_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrint_s(a: v8f32) -> v8f32 {
    __lasx_xvfrint_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrint_d(a: v4f64) -> v4f64 {
    __lasx_xvfrint_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrsqrt_s(a: v8f32) -> v8f32 {
    __lasx_xvfrsqrt_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrsqrt_d(a: v4f64) -> v4f64 {
    __lasx_xvfrsqrt_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvflogb_s(a: v8f32) -> v8f32 {
    __lasx_xvflogb_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvflogb_d(a: v4f64) -> v4f64 {
    __lasx_xvflogb_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvth_s_h(a: v16i16) -> v8f32 {
    __lasx_xvfcvth_s_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvth_d_s(a: v8f32) -> v4f64 {
    __lasx_xvfcvth_d_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvtl_s_h(a: v16i16) -> v8f32 {
    __lasx_xvfcvtl_s_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcvtl_d_s(a: v8f32) -> v4f64 {
    __lasx_xvfcvtl_d_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftint_w_s(a: v8f32) -> v8i32 {
    __lasx_xvftint_w_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftint_l_d(a: v4f64) -> v4i64 {
    __lasx_xvftint_l_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftint_wu_s(a: v8f32) -> v8u32 {
    __lasx_xvftint_wu_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftint_lu_d(a: v4f64) -> v4u64 {
    __lasx_xvftint_lu_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrz_w_s(a: v8f32) -> v8i32 {
    __lasx_xvftintrz_w_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrz_l_d(a: v4f64) -> v4i64 {
    __lasx_xvftintrz_l_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrz_wu_s(a: v8f32) -> v8u32 {
    __lasx_xvftintrz_wu_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrz_lu_d(a: v4f64) -> v4u64 {
    __lasx_xvftintrz_lu_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffint_s_w(a: v8i32) -> v8f32 {
    __lasx_xvffint_s_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffint_d_l(a: v4i64) -> v4f64 {
    __lasx_xvffint_d_l(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffint_s_wu(a: v8u32) -> v8f32 {
    __lasx_xvffint_s_wu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffint_d_lu(a: v4u64) -> v4f64 {
    __lasx_xvffint_d_lu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve_b(a: v32i8, b: i32) -> v32i8 {
    __lasx_xvreplve_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve_h(a: v16i16, b: i32) -> v16i16 {
    __lasx_xvreplve_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve_w(a: v8i32, b: i32) -> v8i32 {
    __lasx_xvreplve_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve_d(a: v4i64, b: i32) -> v4i64 {
    __lasx_xvreplve_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpermi_w<const IMM8: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvpermi_w(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvandn_v(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvandn_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvneg_b(a: v32i8) -> v32i8 {
    __lasx_xvneg_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvneg_h(a: v16i16) -> v16i16 {
    __lasx_xvneg_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvneg_w(a: v8i32) -> v8i32 {
    __lasx_xvneg_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvneg_d(a: v4i64) -> v4i64 {
    __lasx_xvneg_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvmuh_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvmuh_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvmuh_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmuh_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_bu(a: v32u8, b: v32u8) -> v32u8 {
    __lasx_xvmuh_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_hu(a: v16u16, b: v16u16) -> v16u16 {
    __lasx_xvmuh_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_wu(a: v8u32, b: v8u32) -> v8u32 {
    __lasx_xvmuh_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmuh_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvmuh_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_h_b<const IMM3: u32>(a: v32i8) -> v16i16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsllwil_h_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_w_h<const IMM4: u32>(a: v16i16) -> v8i32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsllwil_w_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_d_w<const IMM5: u32>(a: v8i32) -> v4i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsllwil_d_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_hu_bu<const IMM3: u32>(a: v32u8) -> v16u16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvsllwil_hu_bu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_wu_hu<const IMM4: u32>(a: v16u16) -> v8u32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsllwil_wu_hu(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsllwil_du_wu<const IMM5: u32>(a: v8u32) -> v4u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsllwil_du_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsran_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvsran_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsran_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvsran_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsran_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvsran_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvssran_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvssran_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvssran_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_bu_h(a: v16u16, b: v16u16) -> v32u8 {
    __lasx_xvssran_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_hu_w(a: v8u32, b: v8u32) -> v16u16 {
    __lasx_xvssran_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssran_wu_d(a: v4u64, b: v4u64) -> v8u32 {
    __lasx_xvssran_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarn_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvsrarn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarn_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvsrarn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarn_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvsrarn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvssrarn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvssrarn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvssrarn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_bu_h(a: v16u16, b: v16u16) -> v32u8 {
    __lasx_xvssrarn_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_hu_w(a: v8u32, b: v8u32) -> v16u16 {
    __lasx_xvssrarn_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarn_wu_d(a: v4u64, b: v4u64) -> v8u32 {
    __lasx_xvssrarn_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrln_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvsrln_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrln_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvsrln_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrln_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvsrln_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_bu_h(a: v16u16, b: v16u16) -> v32u8 {
    __lasx_xvssrln_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_hu_w(a: v8u32, b: v8u32) -> v16u16 {
    __lasx_xvssrln_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_wu_d(a: v4u64, b: v4u64) -> v8u32 {
    __lasx_xvssrln_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrn_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvsrlrn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrn_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvsrlrn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrn_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvsrlrn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_bu_h(a: v16u16, b: v16u16) -> v32u8 {
    __lasx_xvssrlrn_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_hu_w(a: v8u32, b: v8u32) -> v16u16 {
    __lasx_xvssrlrn_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_wu_d(a: v4u64, b: v4u64) -> v8u32 {
    __lasx_xvssrlrn_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrstpi_b<const IMM5: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvfrstpi_b(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrstpi_h<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvfrstpi_h(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrstp_b(a: v32i8, b: v32i8, c: v32i8) -> v32i8 {
    __lasx_xvfrstp_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrstp_h(a: v16i16, b: v16i16, c: v16i16) -> v16i16 {
    __lasx_xvfrstp_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvshuf4i_d<const IMM8: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvshuf4i_d(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbsrl_v<const IMM5: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvbsrl_v(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvbsll_v<const IMM5: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvbsll_v(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextrins_b<const IMM8: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvextrins_b(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextrins_h<const IMM8: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvextrins_h(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextrins_w<const IMM8: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvextrins_w(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextrins_d<const IMM8: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvextrins_d(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmskltz_b(a: v32i8) -> v32i8 {
    __lasx_xvmskltz_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmskltz_h(a: v16i16) -> v16i16 {
    __lasx_xvmskltz_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmskltz_w(a: v8i32) -> v8i32 {
    __lasx_xvmskltz_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmskltz_d(a: v4i64) -> v4i64 {
    __lasx_xvmskltz_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsigncov_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvsigncov_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsigncov_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvsigncov_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsigncov_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvsigncov_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsigncov_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsigncov_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmadd_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32 {
    __lasx_xvfmadd_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmadd_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64 {
    __lasx_xvfmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmsub_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32 {
    __lasx_xvfmsub_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfmsub_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64 {
    __lasx_xvfmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfnmadd_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32 {
    __lasx_xvfnmadd_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfnmadd_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64 {
    __lasx_xvfnmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfnmsub_s(a: v8f32, b: v8f32, c: v8f32) -> v8f32 {
    __lasx_xvfnmsub_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfnmsub_d(a: v4f64, b: v4f64, c: v4f64) -> v4f64 {
    __lasx_xvfnmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrne_w_s(a: v8f32) -> v8i32 {
    __lasx_xvftintrne_w_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrne_l_d(a: v4f64) -> v4i64 {
    __lasx_xvftintrne_l_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrp_w_s(a: v8f32) -> v8i32 {
    __lasx_xvftintrp_w_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrp_l_d(a: v4f64) -> v4i64 {
    __lasx_xvftintrp_l_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrm_w_s(a: v8f32) -> v8i32 {
    __lasx_xvftintrm_w_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrm_l_d(a: v4f64) -> v4i64 {
    __lasx_xvftintrm_l_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftint_w_d(a: v4f64, b: v4f64) -> v8i32 {
    __lasx_xvftint_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffint_s_l(a: v4i64, b: v4i64) -> v8f32 {
    __lasx_xvffint_s_l(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrz_w_d(a: v4f64, b: v4f64) -> v8i32 {
    __lasx_xvftintrz_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrp_w_d(a: v4f64, b: v4f64) -> v8i32 {
    __lasx_xvftintrp_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrm_w_d(a: v4f64, b: v4f64) -> v8i32 {
    __lasx_xvftintrm_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrne_w_d(a: v4f64, b: v4f64) -> v8i32 {
    __lasx_xvftintrne_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftinth_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftinth_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintl_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintl_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffinth_d_w(a: v8i32) -> v4f64 {
    __lasx_xvffinth_d_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvffintl_d_w(a: v8i32) -> v4f64 {
    __lasx_xvffintl_d_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrzh_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrzh_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrzl_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrzl_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrph_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrph_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrpl_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrpl_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrmh_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrmh_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrml_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrml_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrneh_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrneh_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvftintrnel_l_s(a: v8f32) -> v4i64 {
    __lasx_xvftintrnel_l_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrne_s(a: v8f32) -> v8f32 {
    __lasx_xvfrintrne_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrne_d(a: v4f64) -> v4f64 {
    __lasx_xvfrintrne_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrz_s(a: v8f32) -> v8f32 {
    __lasx_xvfrintrz_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrz_d(a: v4f64) -> v4f64 {
    __lasx_xvfrintrz_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrp_s(a: v8f32) -> v8f32 {
    __lasx_xvfrintrp_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrp_d(a: v4f64) -> v4f64 {
    __lasx_xvfrintrp_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrm_s(a: v8f32) -> v8f32 {
    __lasx_xvfrintrm_s(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfrintrm_d(a: v4f64) -> v4f64 {
    __lasx_xvfrintrm_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvld<const IMM_S12: i32>(mem_addr: *const i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S12, 12);
    __lasx_xvld(mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvst<const IMM_S12: i32>(a: v32i8, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S12, 12);
    __lasx_xvst(a, mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_b<const IMM_S8: i32, const IMM4: u32>(a: v32i8, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvstelm_b(a, mem_addr, IMM_S8, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_h<const IMM_S8: i32, const IMM3: u32>(a: v16i16, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvstelm_h(a, mem_addr, IMM_S8, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_w<const IMM_S8: i32, const IMM2: u32>(a: v8i32, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvstelm_w(a, mem_addr, IMM_S8, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_d<const IMM_S8: i32, const IMM1: u32>(a: v4i64, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM1, 1);
    __lasx_xvstelm_d(a, mem_addr, IMM_S8, IMM1)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvinsve0_w<const IMM3: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvinsve0_w(a, b, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvinsve0_d<const IMM2: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvinsve0_d(a, b, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve_w<const IMM3: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvpickve_w(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve_d<const IMM2: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvpickve_d(a, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvssrlrn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvssrlrn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrn_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvssrlrn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_b_h(a: v16i16, b: v16i16) -> v32i8 {
    __lasx_xvssrln_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_h_w(a: v8i32, b: v8i32) -> v16i16 {
    __lasx_xvssrln_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrln_w_d(a: v4i64, b: v4i64) -> v8i32 {
    __lasx_xvssrln_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvorn_v(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvorn_v(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldi<const IMM_S13: i32>() -> v4i64 {
    static_assert_simm_bits!(IMM_S13, 13);
    __lasx_xvldi(IMM_S13)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldx(mem_addr: *const i8, b: i64) -> v32i8 {
    __lasx_xvldx(mem_addr, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstx(a: v32i8, mem_addr: *mut i8, b: i64) {
    __lasx_xvstx(a, mem_addr, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextl_qu_du(a: v4u64) -> v4u64 {
    __lasx_xvextl_qu_du(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvinsgr2vr_w<const IMM3: u32>(a: v8i32, b: i32) -> v8i32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvinsgr2vr_w(a, b, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvinsgr2vr_d<const IMM2: u32>(a: v4i64, b: i64) -> v4i64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvinsgr2vr_d(a, b, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve0_b(a: v32i8) -> v32i8 {
    __lasx_xvreplve0_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve0_h(a: v16i16) -> v16i16 {
    __lasx_xvreplve0_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve0_w(a: v8i32) -> v8i32 {
    __lasx_xvreplve0_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve0_d(a: v4i64) -> v4i64 {
    __lasx_xvreplve0_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvreplve0_q(a: v32i8) -> v32i8 {
    __lasx_xvreplve0_q(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_h_b(a: v32i8) -> v16i16 {
    __lasx_vext2xv_h_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_w_h(a: v16i16) -> v8i32 {
    __lasx_vext2xv_w_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_d_w(a: v8i32) -> v4i64 {
    __lasx_vext2xv_d_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_w_b(a: v32i8) -> v8i32 {
    __lasx_vext2xv_w_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_d_h(a: v16i16) -> v4i64 {
    __lasx_vext2xv_d_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_d_b(a: v32i8) -> v4i64 {
    __lasx_vext2xv_d_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_hu_bu(a: v32i8) -> v16i16 {
    __lasx_vext2xv_hu_bu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_wu_hu(a: v16i16) -> v8i32 {
    __lasx_vext2xv_wu_hu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_du_wu(a: v8i32) -> v4i64 {
    __lasx_vext2xv_du_wu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_wu_bu(a: v32i8) -> v8i32 {
    __lasx_vext2xv_wu_bu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_du_hu(a: v16i16) -> v4i64 {
    __lasx_vext2xv_du_hu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_vext2xv_du_bu(a: v32i8) -> v4i64 {
    __lasx_vext2xv_du_bu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpermi_q<const IMM8: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvpermi_q(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpermi_d<const IMM8: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lasx_xvpermi_d(a, IMM8)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvperm_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvperm_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_b<const IMM_S12: i32>(mem_addr: *const i8) -> v32i8 {
    static_assert_simm_bits!(IMM_S12, 12);
    __lasx_xvldrepl_b(mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_h<const IMM_S11: i32>(mem_addr: *const i8) -> v16i16 {
    static_assert_simm_bits!(IMM_S11, 11);
    __lasx_xvldrepl_h(mem_addr, IMM_S11)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_w<const IMM_S10: i32>(mem_addr: *const i8) -> v8i32 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lasx_xvldrepl_w(mem_addr, IMM_S10)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_d<const IMM_S9: i32>(mem_addr: *const i8) -> v4i64 {
    static_assert_simm_bits!(IMM_S9, 9);
    __lasx_xvldrepl_d(mem_addr, IMM_S9)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve2gr_w<const IMM3: u32>(a: v8i32) -> i32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvpickve2gr_w(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve2gr_wu<const IMM3: u32>(a: v8i32) -> u32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvpickve2gr_wu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve2gr_d<const IMM2: u32>(a: v4i64) -> i64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvpickve2gr_d(a, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve2gr_du<const IMM2: u32>(a: v4i64) -> u64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvpickve2gr_du(a, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvaddwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvaddwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvaddwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvaddwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvaddwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvaddwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvaddwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvaddwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsubwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvsubwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvsubwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvsubwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvsubwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvsubwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvsubwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwev_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvsubwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmulwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvmulwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvmulwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvmulwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvmulwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvmulwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvmulwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvmulwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvaddwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvaddwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvaddwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvaddwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvaddwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvaddwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvaddwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvaddwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsubwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvsubwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvsubwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvsubwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvsubwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvsubwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvsubwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsubwod_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvsubwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvmulwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_d_w(a: v8i32, b: v8i32) -> v4i64 {
    __lasx_xvmulwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_w_h(a: v16i16, b: v16i16) -> v8i32 {
    __lasx_xvmulwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_h_b(a: v32i8, b: v32i8) -> v16i16 {
    __lasx_xvmulwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_q_du(a: v4u64, b: v4u64) -> v4i64 {
    __lasx_xvmulwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_d_wu(a: v8u32, b: v8u32) -> v4i64 {
    __lasx_xvmulwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_w_hu(a: v16u16, b: v16u16) -> v8i32 {
    __lasx_xvmulwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_h_bu(a: v32u8, b: v32u8) -> v16i16 {
    __lasx_xvmulwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_d_wu_w(a: v8u32, b: v8i32) -> v4i64 {
    __lasx_xvaddwev_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_w_hu_h(a: v16u16, b: v16i16) -> v8i32 {
    __lasx_xvaddwev_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_h_bu_b(a: v32u8, b: v32i8) -> v16i16 {
    __lasx_xvaddwev_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_d_wu_w(a: v8u32, b: v8i32) -> v4i64 {
    __lasx_xvmulwev_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_w_hu_h(a: v16u16, b: v16i16) -> v8i32 {
    __lasx_xvmulwev_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_h_bu_b(a: v32u8, b: v32i8) -> v16i16 {
    __lasx_xvmulwev_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_d_wu_w(a: v8u32, b: v8i32) -> v4i64 {
    __lasx_xvaddwod_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_w_hu_h(a: v16u16, b: v16i16) -> v8i32 {
    __lasx_xvaddwod_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_h_bu_b(a: v32u8, b: v32i8) -> v16i16 {
    __lasx_xvaddwod_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_d_wu_w(a: v8u32, b: v8i32) -> v4i64 {
    __lasx_xvmulwod_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_w_hu_h(a: v16u16, b: v16i16) -> v8i32 {
    __lasx_xvmulwod_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_h_bu_b(a: v32u8, b: v32i8) -> v16i16 {
    __lasx_xvmulwod_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvhaddw_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhaddw_qu_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvhaddw_qu_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_q_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvhsubw_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvhsubw_qu_du(a: v4u64, b: v4u64) -> v4u64 {
    __lasx_xvhsubw_qu_du(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_q_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    __lasx_xvmaddwev_q_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_d_w(a: v4i64, b: v8i32, c: v8i32) -> v4i64 {
    __lasx_xvmaddwev_d_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_w_h(a: v8i32, b: v16i16, c: v16i16) -> v8i32 {
    __lasx_xvmaddwev_w_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_h_b(a: v16i16, b: v32i8, c: v32i8) -> v16i16 {
    __lasx_xvmaddwev_h_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_q_du(a: v4u64, b: v4u64, c: v4u64) -> v4u64 {
    __lasx_xvmaddwev_q_du(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_d_wu(a: v4u64, b: v8u32, c: v8u32) -> v4u64 {
    __lasx_xvmaddwev_d_wu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_w_hu(a: v8u32, b: v16u16, c: v16u16) -> v8u32 {
    __lasx_xvmaddwev_w_hu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_h_bu(a: v16u16, b: v32u8, c: v32u8) -> v16u16 {
    __lasx_xvmaddwev_h_bu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_q_d(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    __lasx_xvmaddwod_q_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_d_w(a: v4i64, b: v8i32, c: v8i32) -> v4i64 {
    __lasx_xvmaddwod_d_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_w_h(a: v8i32, b: v16i16, c: v16i16) -> v8i32 {
    __lasx_xvmaddwod_w_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_h_b(a: v16i16, b: v32i8, c: v32i8) -> v16i16 {
    __lasx_xvmaddwod_h_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_q_du(a: v4u64, b: v4u64, c: v4u64) -> v4u64 {
    __lasx_xvmaddwod_q_du(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_d_wu(a: v4u64, b: v8u32, c: v8u32) -> v4u64 {
    __lasx_xvmaddwod_d_wu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_w_hu(a: v8u32, b: v16u16, c: v16u16) -> v8u32 {
    __lasx_xvmaddwod_w_hu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_h_bu(a: v16u16, b: v32u8, c: v32u8) -> v16u16 {
    __lasx_xvmaddwod_h_bu(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_q_du_d(a: v4i64, b: v4u64, c: v4i64) -> v4i64 {
    __lasx_xvmaddwev_q_du_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_d_wu_w(a: v4i64, b: v8u32, c: v8i32) -> v4i64 {
    __lasx_xvmaddwev_d_wu_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_w_hu_h(a: v8i32, b: v16u16, c: v16i16) -> v8i32 {
    __lasx_xvmaddwev_w_hu_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwev_h_bu_b(a: v16i16, b: v32u8, c: v32i8) -> v16i16 {
    __lasx_xvmaddwev_h_bu_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_q_du_d(a: v4i64, b: v4u64, c: v4i64) -> v4i64 {
    __lasx_xvmaddwod_q_du_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_d_wu_w(a: v4i64, b: v8u32, c: v8i32) -> v4i64 {
    __lasx_xvmaddwod_d_wu_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_w_hu_h(a: v8i32, b: v16u16, c: v16i16) -> v8i32 {
    __lasx_xvmaddwod_w_hu_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmaddwod_h_bu_b(a: v16i16, b: v32u8, c: v32i8) -> v16i16 {
    __lasx_xvmaddwod_h_bu_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotr_b(a: v32i8, b: v32i8) -> v32i8 {
    __lasx_xvrotr_b(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotr_h(a: v16i16, b: v16i16) -> v16i16 {
    __lasx_xvrotr_h(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotr_w(a: v8i32, b: v8i32) -> v8i32 {
    __lasx_xvrotr_w(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotr_d(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvrotr_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvadd_q(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvadd_q(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsub_q(a: v4i64, b: v4i64) -> v4i64 {
    __lasx_xvsub_q(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwev_q_du_d(a: v4u64, b: v4i64) -> v4i64 {
    __lasx_xvaddwev_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvaddwod_q_du_d(a: v4u64, b: v4i64) -> v4i64 {
    __lasx_xvaddwod_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwev_q_du_d(a: v4u64, b: v4i64) -> v4i64 {
    __lasx_xvmulwev_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmulwod_q_du_d(a: v4u64, b: v4i64) -> v4i64 {
    __lasx_xvmulwod_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmskgez_b(a: v32i8) -> v32i8 {
    __lasx_xvmskgez_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvmsknz_b(a: v32i8) -> v32i8 {
    __lasx_xvmsknz_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_h_b(a: v32i8) -> v16i16 {
    __lasx_xvexth_h_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_w_h(a: v16i16) -> v8i32 {
    __lasx_xvexth_w_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_d_w(a: v8i32) -> v4i64 {
    __lasx_xvexth_d_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_q_d(a: v4i64) -> v4i64 {
    __lasx_xvexth_q_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_hu_bu(a: v32u8) -> v16u16 {
    __lasx_xvexth_hu_bu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_wu_hu(a: v16u16) -> v8u32 {
    __lasx_xvexth_wu_hu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_du_wu(a: v8u32) -> v4u64 {
    __lasx_xvexth_du_wu(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvexth_qu_du(a: v4u64) -> v4u64 {
    __lasx_xvexth_qu_du(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotri_b<const IMM3: u32>(a: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvrotri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotri_h<const IMM4: u32>(a: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvrotri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotri_w<const IMM5: u32>(a: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvrotri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrotri_d<const IMM6: u32>(a: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvrotri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvextl_q_d(a: v4i64) -> v4i64 {
    __lasx_xvextl_q_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrlni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrlni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrlni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvsrlni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrlrni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrlrni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrlrni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrlrni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvsrlrni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrlni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrlni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrlni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrlni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_bu_h<const IMM4: u32>(a: v32u8, b: v32i8) -> v32u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrlni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_hu_w<const IMM5: u32>(a: v16u16, b: v16i16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrlni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_wu_d<const IMM6: u32>(a: v8u32, b: v8i32) -> v8u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrlni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlni_du_q<const IMM7: u32>(a: v4u64, b: v4i64) -> v4u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrlni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrlrni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrlrni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrlrni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrlrni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_bu_h<const IMM4: u32>(a: v32u8, b: v32i8) -> v32u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrlrni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_hu_w<const IMM5: u32>(a: v16u16, b: v16i16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrlrni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_wu_d<const IMM6: u32>(a: v8u32, b: v8i32) -> v8u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrlrni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrlrni_du_q<const IMM7: u32>(a: v4u64, b: v4i64) -> v4u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrlrni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrani_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrani_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrani_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrani_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrani_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrani_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrani_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvsrani_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvsrarni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvsrarni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvsrarni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvsrarni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvsrarni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrani_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrani_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrani_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrani_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_bu_h<const IMM4: u32>(a: v32u8, b: v32i8) -> v32u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrani_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_hu_w<const IMM5: u32>(a: v16u16, b: v16i16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrani_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_wu_d<const IMM6: u32>(a: v8u32, b: v8i32) -> v8u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrani_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrani_du_q<const IMM7: u32>(a: v4u64, b: v4i64) -> v4u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrani_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_b_h<const IMM4: u32>(a: v32i8, b: v32i8) -> v32i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrarni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_h_w<const IMM5: u32>(a: v16i16, b: v16i16) -> v16i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrarni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_w_d<const IMM6: u32>(a: v8i32, b: v8i32) -> v8i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrarni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_d_q<const IMM7: u32>(a: v4i64, b: v4i64) -> v4i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrarni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_bu_h<const IMM4: u32>(a: v32u8, b: v32i8) -> v32u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvssrarni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_hu_w<const IMM5: u32>(a: v16u16, b: v16i16) -> v16u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lasx_xvssrarni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_wu_d<const IMM6: u32>(a: v8u32, b: v8i32) -> v8u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lasx_xvssrarni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvssrarni_du_q<const IMM7: u32>(a: v4u64, b: v4i64) -> v4u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lasx_xvssrarni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbnz_b(a: v32u8) -> i32 {
    __lasx_xbnz_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbnz_d(a: v4u64) -> i32 {
    __lasx_xbnz_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbnz_h(a: v16u16) -> i32 {
    __lasx_xbnz_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbnz_v(a: v32u8) -> i32 {
    __lasx_xbnz_v(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbnz_w(a: v8u32) -> i32 {
    __lasx_xbnz_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbz_b(a: v32u8) -> i32 {
    __lasx_xbz_b(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbz_d(a: v4u64) -> i32 {
    __lasx_xbz_d(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbz_h(a: v16u16) -> i32 {
    __lasx_xbz_h(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbz_v(a: v32u8) -> i32 {
    __lasx_xbz_v(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xbz_w(a: v8u32) -> i32 {
    __lasx_xbz_w(a)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_caf_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_caf_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_caf_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_caf_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_ceq_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_ceq_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_ceq_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_ceq_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cle_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cle_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cle_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cle_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_clt_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_clt_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_clt_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_clt_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cne_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cne_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cne_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cne_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cor_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cor_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cor_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cor_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cueq_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cueq_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cueq_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cueq_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cule_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cule_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cule_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cule_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cult_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cult_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cult_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cult_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cun_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cun_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cune_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_cune_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cune_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cune_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_cun_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_cun_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_saf_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_saf_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_saf_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_saf_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_seq_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_seq_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_seq_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_seq_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sle_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sle_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sle_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sle_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_slt_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_slt_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_slt_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_slt_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sne_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sne_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sne_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sne_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sor_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sor_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sor_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sor_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sueq_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sueq_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sueq_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sueq_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sule_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sule_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sule_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sule_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sult_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sult_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sult_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sult_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sun_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sun_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sune_d(a: v4f64, b: v4f64) -> v4i64 {
    __lasx_xvfcmp_sune_d(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sune_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sune_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvfcmp_sun_s(a: v8f32, b: v8f32) -> v8i32 {
    __lasx_xvfcmp_sun_s(a, b)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve_d_f<const IMM2: u32>(a: v4f64) -> v4f64 {
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvpickve_d_f(a, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvpickve_w_f<const IMM3: u32>(a: v8f32) -> v8f32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvpickve_w_f(a, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepli_b<const IMM_S10: i32>() -> v32i8 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lasx_xvrepli_b(IMM_S10)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepli_d<const IMM_S10: i32>() -> v4i64 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lasx_xvrepli_d(IMM_S10)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepli_h<const IMM_S10: i32>() -> v16i16 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lasx_xvrepli_h(IMM_S10)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvrepli_w<const IMM_S10: i32>() -> v8i32 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lasx_xvrepli_w(IMM_S10)
}
