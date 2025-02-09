// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-loongarch/lsx.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lsx.spec
// ```

use super::types::*;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.lsx.vsll.b"]
    fn __lsx_vsll_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsll.h"]
    fn __lsx_vsll_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsll.w"]
    fn __lsx_vsll_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsll.d"]
    fn __lsx_vsll_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslli.b"]
    fn __lsx_vslli_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslli.h"]
    fn __lsx_vslli_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslli.w"]
    fn __lsx_vslli_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslli.d"]
    fn __lsx_vslli_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsra.b"]
    fn __lsx_vsra_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsra.h"]
    fn __lsx_vsra_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsra.w"]
    fn __lsx_vsra_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsra.d"]
    fn __lsx_vsra_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrai.b"]
    fn __lsx_vsrai_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrai.h"]
    fn __lsx_vsrai_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrai.w"]
    fn __lsx_vsrai_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrai.d"]
    fn __lsx_vsrai_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrar.b"]
    fn __lsx_vsrar_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrar.h"]
    fn __lsx_vsrar_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrar.w"]
    fn __lsx_vsrar_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrar.d"]
    fn __lsx_vsrar_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrari.b"]
    fn __lsx_vsrari_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrari.h"]
    fn __lsx_vsrari_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrari.w"]
    fn __lsx_vsrari_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrari.d"]
    fn __lsx_vsrari_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrl.b"]
    fn __lsx_vsrl_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrl.h"]
    fn __lsx_vsrl_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrl.w"]
    fn __lsx_vsrl_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrl.d"]
    fn __lsx_vsrl_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrli.b"]
    fn __lsx_vsrli_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrli.h"]
    fn __lsx_vsrli_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrli.w"]
    fn __lsx_vsrli_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrli.d"]
    fn __lsx_vsrli_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrlr.b"]
    fn __lsx_vsrlr_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrlr.h"]
    fn __lsx_vsrlr_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrlr.w"]
    fn __lsx_vsrlr_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrlr.d"]
    fn __lsx_vsrlr_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrlri.b"]
    fn __lsx_vsrlri_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrlri.h"]
    fn __lsx_vsrlri_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrlri.w"]
    fn __lsx_vsrlri_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrlri.d"]
    fn __lsx_vsrlri_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vbitclr.b"]
    fn __lsx_vbitclr_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitclr.h"]
    fn __lsx_vbitclr_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitclr.w"]
    fn __lsx_vbitclr_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitclr.d"]
    fn __lsx_vbitclr_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vbitclri.b"]
    fn __lsx_vbitclri_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitclri.h"]
    fn __lsx_vbitclri_h(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitclri.w"]
    fn __lsx_vbitclri_w(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitclri.d"]
    fn __lsx_vbitclri_d(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vbitset.b"]
    fn __lsx_vbitset_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitset.h"]
    fn __lsx_vbitset_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitset.w"]
    fn __lsx_vbitset_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitset.d"]
    fn __lsx_vbitset_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vbitseti.b"]
    fn __lsx_vbitseti_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitseti.h"]
    fn __lsx_vbitseti_h(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitseti.w"]
    fn __lsx_vbitseti_w(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitseti.d"]
    fn __lsx_vbitseti_d(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vbitrev.b"]
    fn __lsx_vbitrev_b(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitrev.h"]
    fn __lsx_vbitrev_h(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitrev.w"]
    fn __lsx_vbitrev_w(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitrev.d"]
    fn __lsx_vbitrev_d(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vbitrevi.b"]
    fn __lsx_vbitrevi_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitrevi.h"]
    fn __lsx_vbitrevi_h(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vbitrevi.w"]
    fn __lsx_vbitrevi_w(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vbitrevi.d"]
    fn __lsx_vbitrevi_d(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vadd.b"]
    fn __lsx_vadd_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vadd.h"]
    fn __lsx_vadd_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vadd.w"]
    fn __lsx_vadd_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vadd.d"]
    fn __lsx_vadd_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddi.bu"]
    fn __lsx_vaddi_bu(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vaddi.hu"]
    fn __lsx_vaddi_hu(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddi.wu"]
    fn __lsx_vaddi_wu(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddi.du"]
    fn __lsx_vaddi_du(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsub.b"]
    fn __lsx_vsub_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsub.h"]
    fn __lsx_vsub_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsub.w"]
    fn __lsx_vsub_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsub.d"]
    fn __lsx_vsub_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubi.bu"]
    fn __lsx_vsubi_bu(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsubi.hu"]
    fn __lsx_vsubi_hu(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsubi.wu"]
    fn __lsx_vsubi_wu(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsubi.du"]
    fn __lsx_vsubi_du(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmax.b"]
    fn __lsx_vmax_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmax.h"]
    fn __lsx_vmax_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmax.w"]
    fn __lsx_vmax_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmax.d"]
    fn __lsx_vmax_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaxi.b"]
    fn __lsx_vmaxi_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmaxi.h"]
    fn __lsx_vmaxi_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmaxi.w"]
    fn __lsx_vmaxi_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmaxi.d"]
    fn __lsx_vmaxi_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmax.bu"]
    fn __lsx_vmax_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmax.hu"]
    fn __lsx_vmax_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmax.wu"]
    fn __lsx_vmax_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmax.du"]
    fn __lsx_vmax_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaxi.bu"]
    fn __lsx_vmaxi_bu(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmaxi.hu"]
    fn __lsx_vmaxi_hu(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmaxi.wu"]
    fn __lsx_vmaxi_wu(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmaxi.du"]
    fn __lsx_vmaxi_du(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmin.b"]
    fn __lsx_vmin_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmin.h"]
    fn __lsx_vmin_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmin.w"]
    fn __lsx_vmin_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmin.d"]
    fn __lsx_vmin_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmini.b"]
    fn __lsx_vmini_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmini.h"]
    fn __lsx_vmini_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmini.w"]
    fn __lsx_vmini_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmini.d"]
    fn __lsx_vmini_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmin.bu"]
    fn __lsx_vmin_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmin.hu"]
    fn __lsx_vmin_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmin.wu"]
    fn __lsx_vmin_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmin.du"]
    fn __lsx_vmin_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmini.bu"]
    fn __lsx_vmini_bu(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmini.hu"]
    fn __lsx_vmini_hu(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmini.wu"]
    fn __lsx_vmini_wu(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmini.du"]
    fn __lsx_vmini_du(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vseq.b"]
    fn __lsx_vseq_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vseq.h"]
    fn __lsx_vseq_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vseq.w"]
    fn __lsx_vseq_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vseq.d"]
    fn __lsx_vseq_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vseqi.b"]
    fn __lsx_vseqi_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vseqi.h"]
    fn __lsx_vseqi_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vseqi.w"]
    fn __lsx_vseqi_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vseqi.d"]
    fn __lsx_vseqi_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslti.b"]
    fn __lsx_vslti_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslt.b"]
    fn __lsx_vslt_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslt.h"]
    fn __lsx_vslt_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslt.w"]
    fn __lsx_vslt_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslt.d"]
    fn __lsx_vslt_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslti.h"]
    fn __lsx_vslti_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslti.w"]
    fn __lsx_vslti_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslti.d"]
    fn __lsx_vslti_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslt.bu"]
    fn __lsx_vslt_bu(a: v16u8, b: v16u8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslt.hu"]
    fn __lsx_vslt_hu(a: v8u16, b: v8u16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslt.wu"]
    fn __lsx_vslt_wu(a: v4u32, b: v4u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslt.du"]
    fn __lsx_vslt_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslti.bu"]
    fn __lsx_vslti_bu(a: v16u8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslti.hu"]
    fn __lsx_vslti_hu(a: v8u16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslti.wu"]
    fn __lsx_vslti_wu(a: v4u32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslti.du"]
    fn __lsx_vslti_du(a: v2u64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsle.b"]
    fn __lsx_vsle_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsle.h"]
    fn __lsx_vsle_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsle.w"]
    fn __lsx_vsle_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsle.d"]
    fn __lsx_vsle_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslei.b"]
    fn __lsx_vslei_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslei.h"]
    fn __lsx_vslei_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslei.w"]
    fn __lsx_vslei_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslei.d"]
    fn __lsx_vslei_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsle.bu"]
    fn __lsx_vsle_bu(a: v16u8, b: v16u8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsle.hu"]
    fn __lsx_vsle_hu(a: v8u16, b: v8u16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsle.wu"]
    fn __lsx_vsle_wu(a: v4u32, b: v4u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsle.du"]
    fn __lsx_vsle_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vslei.bu"]
    fn __lsx_vslei_bu(a: v16u8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vslei.hu"]
    fn __lsx_vslei_hu(a: v8u16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vslei.wu"]
    fn __lsx_vslei_wu(a: v4u32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vslei.du"]
    fn __lsx_vslei_du(a: v2u64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsat.b"]
    fn __lsx_vsat_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsat.h"]
    fn __lsx_vsat_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsat.w"]
    fn __lsx_vsat_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsat.d"]
    fn __lsx_vsat_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsat.bu"]
    fn __lsx_vsat_bu(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vsat.hu"]
    fn __lsx_vsat_hu(a: v8u16, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vsat.wu"]
    fn __lsx_vsat_wu(a: v4u32, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsat.du"]
    fn __lsx_vsat_du(a: v2u64, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vadda.b"]
    fn __lsx_vadda_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vadda.h"]
    fn __lsx_vadda_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vadda.w"]
    fn __lsx_vadda_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vadda.d"]
    fn __lsx_vadda_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsadd.b"]
    fn __lsx_vsadd_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsadd.h"]
    fn __lsx_vsadd_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsadd.w"]
    fn __lsx_vsadd_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsadd.d"]
    fn __lsx_vsadd_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsadd.bu"]
    fn __lsx_vsadd_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vsadd.hu"]
    fn __lsx_vsadd_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vsadd.wu"]
    fn __lsx_vsadd_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsadd.du"]
    fn __lsx_vsadd_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vavg.b"]
    fn __lsx_vavg_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vavg.h"]
    fn __lsx_vavg_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vavg.w"]
    fn __lsx_vavg_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vavg.d"]
    fn __lsx_vavg_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vavg.bu"]
    fn __lsx_vavg_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vavg.hu"]
    fn __lsx_vavg_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vavg.wu"]
    fn __lsx_vavg_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vavg.du"]
    fn __lsx_vavg_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vavgr.b"]
    fn __lsx_vavgr_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vavgr.h"]
    fn __lsx_vavgr_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vavgr.w"]
    fn __lsx_vavgr_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vavgr.d"]
    fn __lsx_vavgr_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vavgr.bu"]
    fn __lsx_vavgr_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vavgr.hu"]
    fn __lsx_vavgr_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vavgr.wu"]
    fn __lsx_vavgr_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vavgr.du"]
    fn __lsx_vavgr_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vssub.b"]
    fn __lsx_vssub_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssub.h"]
    fn __lsx_vssub_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssub.w"]
    fn __lsx_vssub_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssub.d"]
    fn __lsx_vssub_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssub.bu"]
    fn __lsx_vssub_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssub.hu"]
    fn __lsx_vssub_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssub.wu"]
    fn __lsx_vssub_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vssub.du"]
    fn __lsx_vssub_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vabsd.b"]
    fn __lsx_vabsd_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vabsd.h"]
    fn __lsx_vabsd_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vabsd.w"]
    fn __lsx_vabsd_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vabsd.d"]
    fn __lsx_vabsd_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vabsd.bu"]
    fn __lsx_vabsd_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vabsd.hu"]
    fn __lsx_vabsd_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vabsd.wu"]
    fn __lsx_vabsd_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vabsd.du"]
    fn __lsx_vabsd_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmul.b"]
    fn __lsx_vmul_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmul.h"]
    fn __lsx_vmul_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmul.w"]
    fn __lsx_vmul_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmul.d"]
    fn __lsx_vmul_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmadd.b"]
    fn __lsx_vmadd_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmadd.h"]
    fn __lsx_vmadd_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmadd.w"]
    fn __lsx_vmadd_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmadd.d"]
    fn __lsx_vmadd_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmsub.b"]
    fn __lsx_vmsub_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmsub.h"]
    fn __lsx_vmsub_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmsub.w"]
    fn __lsx_vmsub_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmsub.d"]
    fn __lsx_vmsub_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vdiv.b"]
    fn __lsx_vdiv_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vdiv.h"]
    fn __lsx_vdiv_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vdiv.w"]
    fn __lsx_vdiv_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vdiv.d"]
    fn __lsx_vdiv_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vdiv.bu"]
    fn __lsx_vdiv_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vdiv.hu"]
    fn __lsx_vdiv_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vdiv.wu"]
    fn __lsx_vdiv_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vdiv.du"]
    fn __lsx_vdiv_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vhaddw.h.b"]
    fn __lsx_vhaddw_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vhaddw.w.h"]
    fn __lsx_vhaddw_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vhaddw.d.w"]
    fn __lsx_vhaddw_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vhaddw.hu.bu"]
    fn __lsx_vhaddw_hu_bu(a: v16u8, b: v16u8) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vhaddw.wu.hu"]
    fn __lsx_vhaddw_wu_hu(a: v8u16, b: v8u16) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vhaddw.du.wu"]
    fn __lsx_vhaddw_du_wu(a: v4u32, b: v4u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vhsubw.h.b"]
    fn __lsx_vhsubw_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vhsubw.w.h"]
    fn __lsx_vhsubw_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vhsubw.d.w"]
    fn __lsx_vhsubw_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vhsubw.hu.bu"]
    fn __lsx_vhsubw_hu_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vhsubw.wu.hu"]
    fn __lsx_vhsubw_wu_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vhsubw.du.wu"]
    fn __lsx_vhsubw_du_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmod.b"]
    fn __lsx_vmod_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmod.h"]
    fn __lsx_vmod_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmod.w"]
    fn __lsx_vmod_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmod.d"]
    fn __lsx_vmod_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmod.bu"]
    fn __lsx_vmod_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmod.hu"]
    fn __lsx_vmod_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmod.wu"]
    fn __lsx_vmod_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmod.du"]
    fn __lsx_vmod_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vreplve.b"]
    fn __lsx_vreplve_b(a: v16i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vreplve.h"]
    fn __lsx_vreplve_h(a: v8i16, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vreplve.w"]
    fn __lsx_vreplve_w(a: v4i32, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vreplve.d"]
    fn __lsx_vreplve_d(a: v2i64, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vreplvei.b"]
    fn __lsx_vreplvei_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vreplvei.h"]
    fn __lsx_vreplvei_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vreplvei.w"]
    fn __lsx_vreplvei_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vreplvei.d"]
    fn __lsx_vreplvei_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpickev.b"]
    fn __lsx_vpickev_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vpickev.h"]
    fn __lsx_vpickev_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vpickev.w"]
    fn __lsx_vpickev_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vpickev.d"]
    fn __lsx_vpickev_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpickod.b"]
    fn __lsx_vpickod_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vpickod.h"]
    fn __lsx_vpickod_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vpickod.w"]
    fn __lsx_vpickod_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vpickod.d"]
    fn __lsx_vpickod_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vilvh.b"]
    fn __lsx_vilvh_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vilvh.h"]
    fn __lsx_vilvh_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vilvh.w"]
    fn __lsx_vilvh_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vilvh.d"]
    fn __lsx_vilvh_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vilvl.b"]
    fn __lsx_vilvl_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vilvl.h"]
    fn __lsx_vilvl_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vilvl.w"]
    fn __lsx_vilvl_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vilvl.d"]
    fn __lsx_vilvl_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpackev.b"]
    fn __lsx_vpackev_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vpackev.h"]
    fn __lsx_vpackev_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vpackev.w"]
    fn __lsx_vpackev_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vpackev.d"]
    fn __lsx_vpackev_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpackod.b"]
    fn __lsx_vpackod_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vpackod.h"]
    fn __lsx_vpackod_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vpackod.w"]
    fn __lsx_vpackod_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vpackod.d"]
    fn __lsx_vpackod_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vshuf.h"]
    fn __lsx_vshuf_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vshuf.w"]
    fn __lsx_vshuf_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vshuf.d"]
    fn __lsx_vshuf_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vand.v"]
    fn __lsx_vand_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vandi.b"]
    fn __lsx_vandi_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vor.v"]
    fn __lsx_vor_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vori.b"]
    fn __lsx_vori_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vnor.v"]
    fn __lsx_vnor_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vnori.b"]
    fn __lsx_vnori_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vxor.v"]
    fn __lsx_vxor_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vxori.b"]
    fn __lsx_vxori_b(a: v16u8, b: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitsel.v"]
    fn __lsx_vbitsel_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vbitseli.b"]
    fn __lsx_vbitseli_b(a: v16u8, b: v16u8, c: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vshuf4i.b"]
    fn __lsx_vshuf4i_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vshuf4i.h"]
    fn __lsx_vshuf4i_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vshuf4i.w"]
    fn __lsx_vshuf4i_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vreplgr2vr.b"]
    fn __lsx_vreplgr2vr_b(a: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vreplgr2vr.h"]
    fn __lsx_vreplgr2vr_h(a: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vreplgr2vr.w"]
    fn __lsx_vreplgr2vr_w(a: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vreplgr2vr.d"]
    fn __lsx_vreplgr2vr_d(a: i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpcnt.b"]
    fn __lsx_vpcnt_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vpcnt.h"]
    fn __lsx_vpcnt_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vpcnt.w"]
    fn __lsx_vpcnt_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vpcnt.d"]
    fn __lsx_vpcnt_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vclo.b"]
    fn __lsx_vclo_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vclo.h"]
    fn __lsx_vclo_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vclo.w"]
    fn __lsx_vclo_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vclo.d"]
    fn __lsx_vclo_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vclz.b"]
    fn __lsx_vclz_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vclz.h"]
    fn __lsx_vclz_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vclz.w"]
    fn __lsx_vclz_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vclz.d"]
    fn __lsx_vclz_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.b"]
    fn __lsx_vpickve2gr_b(a: v16i8, b: u32) -> i32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.h"]
    fn __lsx_vpickve2gr_h(a: v8i16, b: u32) -> i32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.w"]
    fn __lsx_vpickve2gr_w(a: v4i32, b: u32) -> i32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.d"]
    fn __lsx_vpickve2gr_d(a: v2i64, b: u32) -> i64;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.bu"]
    fn __lsx_vpickve2gr_bu(a: v16i8, b: u32) -> u32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.hu"]
    fn __lsx_vpickve2gr_hu(a: v8i16, b: u32) -> u32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.wu"]
    fn __lsx_vpickve2gr_wu(a: v4i32, b: u32) -> u32;
    #[link_name = "llvm.loongarch.lsx.vpickve2gr.du"]
    fn __lsx_vpickve2gr_du(a: v2i64, b: u32) -> u64;
    #[link_name = "llvm.loongarch.lsx.vinsgr2vr.b"]
    fn __lsx_vinsgr2vr_b(a: v16i8, b: i32, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vinsgr2vr.h"]
    fn __lsx_vinsgr2vr_h(a: v8i16, b: i32, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vinsgr2vr.w"]
    fn __lsx_vinsgr2vr_w(a: v4i32, b: i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vinsgr2vr.d"]
    fn __lsx_vinsgr2vr_d(a: v2i64, b: i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfadd.s"]
    fn __lsx_vfadd_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfadd.d"]
    fn __lsx_vfadd_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfsub.s"]
    fn __lsx_vfsub_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfsub.d"]
    fn __lsx_vfsub_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfmul.s"]
    fn __lsx_vfmul_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmul.d"]
    fn __lsx_vfmul_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfdiv.s"]
    fn __lsx_vfdiv_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfdiv.d"]
    fn __lsx_vfdiv_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfcvt.h.s"]
    fn __lsx_vfcvt_h_s(a: v4f32, b: v4f32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vfcvt.s.d"]
    fn __lsx_vfcvt_s_d(a: v2f64, b: v2f64) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmin.s"]
    fn __lsx_vfmin_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmin.d"]
    fn __lsx_vfmin_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfmina.s"]
    fn __lsx_vfmina_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmina.d"]
    fn __lsx_vfmina_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfmax.s"]
    fn __lsx_vfmax_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmax.d"]
    fn __lsx_vfmax_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfmaxa.s"]
    fn __lsx_vfmaxa_s(a: v4f32, b: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmaxa.d"]
    fn __lsx_vfmaxa_d(a: v2f64, b: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfclass.s"]
    fn __lsx_vfclass_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfclass.d"]
    fn __lsx_vfclass_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfsqrt.s"]
    fn __lsx_vfsqrt_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfsqrt.d"]
    fn __lsx_vfsqrt_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrecip.s"]
    fn __lsx_vfrecip_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrecip.d"]
    fn __lsx_vfrecip_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrecipe.s"]
    fn __lsx_vfrecipe_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrecipe.d"]
    fn __lsx_vfrecipe_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrsqrte.s"]
    fn __lsx_vfrsqrte_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrsqrte.d"]
    fn __lsx_vfrsqrte_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrint.s"]
    fn __lsx_vfrint_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrint.d"]
    fn __lsx_vfrint_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrsqrt.s"]
    fn __lsx_vfrsqrt_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrsqrt.d"]
    fn __lsx_vfrsqrt_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vflogb.s"]
    fn __lsx_vflogb_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vflogb.d"]
    fn __lsx_vflogb_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfcvth.s.h"]
    fn __lsx_vfcvth_s_h(a: v8i16) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfcvth.d.s"]
    fn __lsx_vfcvth_d_s(a: v4f32) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfcvtl.s.h"]
    fn __lsx_vfcvtl_s_h(a: v8i16) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfcvtl.d.s"]
    fn __lsx_vfcvtl_d_s(a: v4f32) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vftint.w.s"]
    fn __lsx_vftint_w_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftint.l.d"]
    fn __lsx_vftint_l_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftint.wu.s"]
    fn __lsx_vftint_wu_s(a: v4f32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vftint.lu.d"]
    fn __lsx_vftint_lu_d(a: v2f64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vftintrz.w.s"]
    fn __lsx_vftintrz_w_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrz.l.d"]
    fn __lsx_vftintrz_l_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrz.wu.s"]
    fn __lsx_vftintrz_wu_s(a: v4f32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vftintrz.lu.d"]
    fn __lsx_vftintrz_lu_d(a: v2f64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vffint.s.w"]
    fn __lsx_vffint_s_w(a: v4i32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vffint.d.l"]
    fn __lsx_vffint_d_l(a: v2i64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vffint.s.wu"]
    fn __lsx_vffint_s_wu(a: v4u32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vffint.d.lu"]
    fn __lsx_vffint_d_lu(a: v2u64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vandn.v"]
    fn __lsx_vandn_v(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vneg.b"]
    fn __lsx_vneg_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vneg.h"]
    fn __lsx_vneg_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vneg.w"]
    fn __lsx_vneg_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vneg.d"]
    fn __lsx_vneg_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmuh.b"]
    fn __lsx_vmuh_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmuh.h"]
    fn __lsx_vmuh_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmuh.w"]
    fn __lsx_vmuh_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmuh.d"]
    fn __lsx_vmuh_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmuh.bu"]
    fn __lsx_vmuh_bu(a: v16u8, b: v16u8) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vmuh.hu"]
    fn __lsx_vmuh_hu(a: v8u16, b: v8u16) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmuh.wu"]
    fn __lsx_vmuh_wu(a: v4u32, b: v4u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmuh.du"]
    fn __lsx_vmuh_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vsllwil.h.b"]
    fn __lsx_vsllwil_h_b(a: v16i8, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsllwil.w.h"]
    fn __lsx_vsllwil_w_h(a: v8i16, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsllwil.d.w"]
    fn __lsx_vsllwil_d_w(a: v4i32, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsllwil.hu.bu"]
    fn __lsx_vsllwil_hu_bu(a: v16u8, b: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vsllwil.wu.hu"]
    fn __lsx_vsllwil_wu_hu(a: v8u16, b: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsllwil.du.wu"]
    fn __lsx_vsllwil_du_wu(a: v4u32, b: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vsran.b.h"]
    fn __lsx_vsran_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsran.h.w"]
    fn __lsx_vsran_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsran.w.d"]
    fn __lsx_vsran_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssran.b.h"]
    fn __lsx_vssran_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssran.h.w"]
    fn __lsx_vssran_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssran.w.d"]
    fn __lsx_vssran_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssran.bu.h"]
    fn __lsx_vssran_bu_h(a: v8u16, b: v8u16) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssran.hu.w"]
    fn __lsx_vssran_hu_w(a: v4u32, b: v4u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssran.wu.d"]
    fn __lsx_vssran_wu_d(a: v2u64, b: v2u64) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsrarn.b.h"]
    fn __lsx_vsrarn_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrarn.h.w"]
    fn __lsx_vsrarn_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrarn.w.d"]
    fn __lsx_vsrarn_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrarn.b.h"]
    fn __lsx_vssrarn_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrarn.h.w"]
    fn __lsx_vssrarn_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrarn.w.d"]
    fn __lsx_vssrarn_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrarn.bu.h"]
    fn __lsx_vssrarn_bu_h(a: v8u16, b: v8u16) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrarn.hu.w"]
    fn __lsx_vssrarn_hu_w(a: v4u32, b: v4u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrarn.wu.d"]
    fn __lsx_vssrarn_wu_d(a: v2u64, b: v2u64) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsrln.b.h"]
    fn __lsx_vsrln_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrln.h.w"]
    fn __lsx_vsrln_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrln.w.d"]
    fn __lsx_vsrln_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrln.bu.h"]
    fn __lsx_vssrln_bu_h(a: v8u16, b: v8u16) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrln.hu.w"]
    fn __lsx_vssrln_hu_w(a: v4u32, b: v4u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrln.wu.d"]
    fn __lsx_vssrln_wu_d(a: v2u64, b: v2u64) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vsrlrn.b.h"]
    fn __lsx_vsrlrn_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrlrn.h.w"]
    fn __lsx_vsrlrn_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrlrn.w.d"]
    fn __lsx_vsrlrn_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrlrn.bu.h"]
    fn __lsx_vssrlrn_bu_h(a: v8u16, b: v8u16) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrlrn.hu.w"]
    fn __lsx_vssrlrn_hu_w(a: v4u32, b: v4u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrlrn.wu.d"]
    fn __lsx_vssrlrn_wu_d(a: v2u64, b: v2u64) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vfrstpi.b"]
    fn __lsx_vfrstpi_b(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vfrstpi.h"]
    fn __lsx_vfrstpi_h(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vfrstp.b"]
    fn __lsx_vfrstp_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vfrstp.h"]
    fn __lsx_vfrstp_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vshuf4i.d"]
    fn __lsx_vshuf4i_d(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vbsrl.v"]
    fn __lsx_vbsrl_v(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vbsll.v"]
    fn __lsx_vbsll_v(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vextrins.b"]
    fn __lsx_vextrins_b(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vextrins.h"]
    fn __lsx_vextrins_h(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vextrins.w"]
    fn __lsx_vextrins_w(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vextrins.d"]
    fn __lsx_vextrins_d(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmskltz.b"]
    fn __lsx_vmskltz_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmskltz.h"]
    fn __lsx_vmskltz_h(a: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmskltz.w"]
    fn __lsx_vmskltz_w(a: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmskltz.d"]
    fn __lsx_vmskltz_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsigncov.b"]
    fn __lsx_vsigncov_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsigncov.h"]
    fn __lsx_vsigncov_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsigncov.w"]
    fn __lsx_vsigncov_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsigncov.d"]
    fn __lsx_vsigncov_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfmadd.s"]
    fn __lsx_vfmadd_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmadd.d"]
    fn __lsx_vfmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfmsub.s"]
    fn __lsx_vfmsub_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfmsub.d"]
    fn __lsx_vfmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfnmadd.s"]
    fn __lsx_vfnmadd_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfnmadd.d"]
    fn __lsx_vfnmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfnmsub.s"]
    fn __lsx_vfnmsub_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfnmsub.d"]
    fn __lsx_vfnmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vftintrne.w.s"]
    fn __lsx_vftintrne_w_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrne.l.d"]
    fn __lsx_vftintrne_l_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrp.w.s"]
    fn __lsx_vftintrp_w_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrp.l.d"]
    fn __lsx_vftintrp_l_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrm.w.s"]
    fn __lsx_vftintrm_w_s(a: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrm.l.d"]
    fn __lsx_vftintrm_l_d(a: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftint.w.d"]
    fn __lsx_vftint_w_d(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vffint.s.l"]
    fn __lsx_vffint_s_l(a: v2i64, b: v2i64) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vftintrz.w.d"]
    fn __lsx_vftintrz_w_d(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrp.w.d"]
    fn __lsx_vftintrp_w_d(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrm.w.d"]
    fn __lsx_vftintrm_w_d(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintrne.w.d"]
    fn __lsx_vftintrne_w_d(a: v2f64, b: v2f64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vftintl.l.s"]
    fn __lsx_vftintl_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftinth.l.s"]
    fn __lsx_vftinth_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vffinth.d.w"]
    fn __lsx_vffinth_d_w(a: v4i32) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vffintl.d.w"]
    fn __lsx_vffintl_d_w(a: v4i32) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vftintrzl.l.s"]
    fn __lsx_vftintrzl_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrzh.l.s"]
    fn __lsx_vftintrzh_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrpl.l.s"]
    fn __lsx_vftintrpl_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrph.l.s"]
    fn __lsx_vftintrph_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrml.l.s"]
    fn __lsx_vftintrml_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrmh.l.s"]
    fn __lsx_vftintrmh_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrnel.l.s"]
    fn __lsx_vftintrnel_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vftintrneh.l.s"]
    fn __lsx_vftintrneh_l_s(a: v4f32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfrintrne.s"]
    fn __lsx_vfrintrne_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrintrne.d"]
    fn __lsx_vfrintrne_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrintrz.s"]
    fn __lsx_vfrintrz_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrintrz.d"]
    fn __lsx_vfrintrz_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrintrp.s"]
    fn __lsx_vfrintrp_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrintrp.d"]
    fn __lsx_vfrintrp_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vfrintrm.s"]
    fn __lsx_vfrintrm_s(a: v4f32) -> v4f32;
    #[link_name = "llvm.loongarch.lsx.vfrintrm.d"]
    fn __lsx_vfrintrm_d(a: v2f64) -> v2f64;
    #[link_name = "llvm.loongarch.lsx.vstelm.b"]
    fn __lsx_vstelm_b(a: v16i8, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lsx.vstelm.h"]
    fn __lsx_vstelm_h(a: v8i16, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lsx.vstelm.w"]
    fn __lsx_vstelm_w(a: v4i32, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lsx.vstelm.d"]
    fn __lsx_vstelm_d(a: v2i64, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lsx.vaddwev.d.w"]
    fn __lsx_vaddwev_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwev.w.h"]
    fn __lsx_vaddwev_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwev.h.b"]
    fn __lsx_vaddwev_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwod.d.w"]
    fn __lsx_vaddwod_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.w.h"]
    fn __lsx_vaddwod_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwod.h.b"]
    fn __lsx_vaddwod_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwev.d.wu"]
    fn __lsx_vaddwev_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwev.w.hu"]
    fn __lsx_vaddwev_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwev.h.bu"]
    fn __lsx_vaddwev_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwod.d.wu"]
    fn __lsx_vaddwod_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.w.hu"]
    fn __lsx_vaddwod_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwod.h.bu"]
    fn __lsx_vaddwod_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwev.d.wu.w"]
    fn __lsx_vaddwev_d_wu_w(a: v4u32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwev.w.hu.h"]
    fn __lsx_vaddwev_w_hu_h(a: v8u16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwev.h.bu.b"]
    fn __lsx_vaddwev_h_bu_b(a: v16u8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwod.d.wu.w"]
    fn __lsx_vaddwod_d_wu_w(a: v4u32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.w.hu.h"]
    fn __lsx_vaddwod_w_hu_h(a: v8u16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vaddwod.h.bu.b"]
    fn __lsx_vaddwod_h_bu_b(a: v16u8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsubwev.d.w"]
    fn __lsx_vsubwev_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwev.w.h"]
    fn __lsx_vsubwev_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsubwev.h.b"]
    fn __lsx_vsubwev_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsubwod.d.w"]
    fn __lsx_vsubwod_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwod.w.h"]
    fn __lsx_vsubwod_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsubwod.h.b"]
    fn __lsx_vsubwod_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsubwev.d.wu"]
    fn __lsx_vsubwev_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwev.w.hu"]
    fn __lsx_vsubwev_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsubwev.h.bu"]
    fn __lsx_vsubwev_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsubwod.d.wu"]
    fn __lsx_vsubwod_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwod.w.hu"]
    fn __lsx_vsubwod_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsubwod.h.bu"]
    fn __lsx_vsubwod_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vaddwev.q.d"]
    fn __lsx_vaddwev_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.q.d"]
    fn __lsx_vaddwod_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwev.q.du"]
    fn __lsx_vaddwev_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.q.du"]
    fn __lsx_vaddwod_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwev.q.d"]
    fn __lsx_vsubwev_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwod.q.d"]
    fn __lsx_vsubwod_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwev.q.du"]
    fn __lsx_vsubwev_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsubwod.q.du"]
    fn __lsx_vsubwod_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwev.q.du.d"]
    fn __lsx_vaddwev_q_du_d(a: v2u64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vaddwod.q.du.d"]
    fn __lsx_vaddwod_q_du_d(a: v2u64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.d.w"]
    fn __lsx_vmulwev_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.w.h"]
    fn __lsx_vmulwev_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwev.h.b"]
    fn __lsx_vmulwev_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwod.d.w"]
    fn __lsx_vmulwod_d_w(a: v4i32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.w.h"]
    fn __lsx_vmulwod_w_h(a: v8i16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwod.h.b"]
    fn __lsx_vmulwod_h_b(a: v16i8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwev.d.wu"]
    fn __lsx_vmulwev_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.w.hu"]
    fn __lsx_vmulwev_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwev.h.bu"]
    fn __lsx_vmulwev_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwod.d.wu"]
    fn __lsx_vmulwod_d_wu(a: v4u32, b: v4u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.w.hu"]
    fn __lsx_vmulwod_w_hu(a: v8u16, b: v8u16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwod.h.bu"]
    fn __lsx_vmulwod_h_bu(a: v16u8, b: v16u8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwev.d.wu.w"]
    fn __lsx_vmulwev_d_wu_w(a: v4u32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.w.hu.h"]
    fn __lsx_vmulwev_w_hu_h(a: v8u16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwev.h.bu.b"]
    fn __lsx_vmulwev_h_bu_b(a: v16u8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwod.d.wu.w"]
    fn __lsx_vmulwod_d_wu_w(a: v4u32, b: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.w.hu.h"]
    fn __lsx_vmulwod_w_hu_h(a: v8u16, b: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmulwod.h.bu.b"]
    fn __lsx_vmulwod_h_bu_b(a: v16u8, b: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmulwev.q.d"]
    fn __lsx_vmulwev_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.q.d"]
    fn __lsx_vmulwod_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.q.du"]
    fn __lsx_vmulwev_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.q.du"]
    fn __lsx_vmulwod_q_du(a: v2u64, b: v2u64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwev.q.du.d"]
    fn __lsx_vmulwev_q_du_d(a: v2u64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmulwod.q.du.d"]
    fn __lsx_vmulwod_q_du_d(a: v2u64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vhaddw.q.d"]
    fn __lsx_vhaddw_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vhaddw.qu.du"]
    fn __lsx_vhaddw_qu_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vhsubw.q.d"]
    fn __lsx_vhsubw_q_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vhsubw.qu.du"]
    fn __lsx_vhsubw_qu_du(a: v2u64, b: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.d.w"]
    fn __lsx_vmaddwev_d_w(a: v2i64, b: v4i32, c: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.w.h"]
    fn __lsx_vmaddwev_w_h(a: v4i32, b: v8i16, c: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.h.b"]
    fn __lsx_vmaddwev_h_b(a: v8i16, b: v16i8, c: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.d.wu"]
    fn __lsx_vmaddwev_d_wu(a: v2u64, b: v4u32, c: v4u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.w.hu"]
    fn __lsx_vmaddwev_w_hu(a: v4u32, b: v8u16, c: v8u16) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.h.bu"]
    fn __lsx_vmaddwev_h_bu(a: v8u16, b: v16u8, c: v16u8) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.d.w"]
    fn __lsx_vmaddwod_d_w(a: v2i64, b: v4i32, c: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.w.h"]
    fn __lsx_vmaddwod_w_h(a: v4i32, b: v8i16, c: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.h.b"]
    fn __lsx_vmaddwod_h_b(a: v8i16, b: v16i8, c: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.d.wu"]
    fn __lsx_vmaddwod_d_wu(a: v2u64, b: v4u32, c: v4u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.w.hu"]
    fn __lsx_vmaddwod_w_hu(a: v4u32, b: v8u16, c: v8u16) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.h.bu"]
    fn __lsx_vmaddwod_h_bu(a: v8u16, b: v16u8, c: v16u8) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.d.wu.w"]
    fn __lsx_vmaddwev_d_wu_w(a: v2i64, b: v4u32, c: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.w.hu.h"]
    fn __lsx_vmaddwev_w_hu_h(a: v4i32, b: v8u16, c: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.h.bu.b"]
    fn __lsx_vmaddwev_h_bu_b(a: v8i16, b: v16u8, c: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.d.wu.w"]
    fn __lsx_vmaddwod_d_wu_w(a: v2i64, b: v4u32, c: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.w.hu.h"]
    fn __lsx_vmaddwod_w_hu_h(a: v4i32, b: v8u16, c: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.h.bu.b"]
    fn __lsx_vmaddwod_h_bu_b(a: v8i16, b: v16u8, c: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.q.d"]
    fn __lsx_vmaddwev_q_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.q.d"]
    fn __lsx_vmaddwod_q_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.q.du"]
    fn __lsx_vmaddwev_q_du(a: v2u64, b: v2u64, c: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.q.du"]
    fn __lsx_vmaddwod_q_du(a: v2u64, b: v2u64, c: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vmaddwev.q.du.d"]
    fn __lsx_vmaddwev_q_du_d(a: v2i64, b: v2u64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmaddwod.q.du.d"]
    fn __lsx_vmaddwod_q_du_d(a: v2i64, b: v2u64, c: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vrotr.b"]
    fn __lsx_vrotr_b(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vrotr.h"]
    fn __lsx_vrotr_h(a: v8i16, b: v8i16) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vrotr.w"]
    fn __lsx_vrotr_w(a: v4i32, b: v4i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vrotr.d"]
    fn __lsx_vrotr_d(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vadd.q"]
    fn __lsx_vadd_q(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsub.q"]
    fn __lsx_vsub_q(a: v2i64, b: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vldrepl.b"]
    fn __lsx_vldrepl_b(a: *const i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vldrepl.h"]
    fn __lsx_vldrepl_h(a: *const i8, b: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vldrepl.w"]
    fn __lsx_vldrepl_w(a: *const i8, b: i32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vldrepl.d"]
    fn __lsx_vldrepl_d(a: *const i8, b: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vmskgez.b"]
    fn __lsx_vmskgez_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vmsknz.b"]
    fn __lsx_vmsknz_b(a: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vexth.h.b"]
    fn __lsx_vexth_h_b(a: v16i8) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vexth.w.h"]
    fn __lsx_vexth_w_h(a: v8i16) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vexth.d.w"]
    fn __lsx_vexth_d_w(a: v4i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vexth.q.d"]
    fn __lsx_vexth_q_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vexth.hu.bu"]
    fn __lsx_vexth_hu_bu(a: v16u8) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vexth.wu.hu"]
    fn __lsx_vexth_wu_hu(a: v8u16) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vexth.du.wu"]
    fn __lsx_vexth_du_wu(a: v4u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vexth.qu.du"]
    fn __lsx_vexth_qu_du(a: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vrotri.b"]
    fn __lsx_vrotri_b(a: v16i8, b: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vrotri.h"]
    fn __lsx_vrotri_h(a: v8i16, b: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vrotri.w"]
    fn __lsx_vrotri_w(a: v4i32, b: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vrotri.d"]
    fn __lsx_vrotri_d(a: v2i64, b: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vextl.q.d"]
    fn __lsx_vextl_q_d(a: v2i64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrlni.b.h"]
    fn __lsx_vsrlni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrlni.h.w"]
    fn __lsx_vsrlni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrlni.w.d"]
    fn __lsx_vsrlni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrlni.d.q"]
    fn __lsx_vsrlni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrlrni.b.h"]
    fn __lsx_vsrlrni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrlrni.h.w"]
    fn __lsx_vsrlrni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrlrni.w.d"]
    fn __lsx_vsrlrni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrlrni.d.q"]
    fn __lsx_vsrlrni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrlni.b.h"]
    fn __lsx_vssrlni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrlni.h.w"]
    fn __lsx_vssrlni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrlni.w.d"]
    fn __lsx_vssrlni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrlni.d.q"]
    fn __lsx_vssrlni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrlni.bu.h"]
    fn __lsx_vssrlni_bu_h(a: v16u8, b: v16i8, c: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrlni.hu.w"]
    fn __lsx_vssrlni_hu_w(a: v8u16, b: v8i16, c: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrlni.wu.d"]
    fn __lsx_vssrlni_wu_d(a: v4u32, b: v4i32, c: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vssrlni.du.q"]
    fn __lsx_vssrlni_du_q(a: v2u64, b: v2i64, c: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.b.h"]
    fn __lsx_vssrlrni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.h.w"]
    fn __lsx_vssrlrni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.w.d"]
    fn __lsx_vssrlrni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.d.q"]
    fn __lsx_vssrlrni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.bu.h"]
    fn __lsx_vssrlrni_bu_h(a: v16u8, b: v16i8, c: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.hu.w"]
    fn __lsx_vssrlrni_hu_w(a: v8u16, b: v8i16, c: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.wu.d"]
    fn __lsx_vssrlrni_wu_d(a: v4u32, b: v4i32, c: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vssrlrni.du.q"]
    fn __lsx_vssrlrni_du_q(a: v2u64, b: v2i64, c: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vsrani.b.h"]
    fn __lsx_vsrani_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrani.h.w"]
    fn __lsx_vsrani_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrani.w.d"]
    fn __lsx_vsrani_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrani.d.q"]
    fn __lsx_vsrani_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vsrarni.b.h"]
    fn __lsx_vsrarni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vsrarni.h.w"]
    fn __lsx_vsrarni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vsrarni.w.d"]
    fn __lsx_vsrarni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vsrarni.d.q"]
    fn __lsx_vsrarni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrani.b.h"]
    fn __lsx_vssrani_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrani.h.w"]
    fn __lsx_vssrani_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrani.w.d"]
    fn __lsx_vssrani_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrani.d.q"]
    fn __lsx_vssrani_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrani.bu.h"]
    fn __lsx_vssrani_bu_h(a: v16u8, b: v16i8, c: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrani.hu.w"]
    fn __lsx_vssrani_hu_w(a: v8u16, b: v8i16, c: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrani.wu.d"]
    fn __lsx_vssrani_wu_d(a: v4u32, b: v4i32, c: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vssrani.du.q"]
    fn __lsx_vssrani_du_q(a: v2u64, b: v2i64, c: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vssrarni.b.h"]
    fn __lsx_vssrarni_b_h(a: v16i8, b: v16i8, c: u32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrarni.h.w"]
    fn __lsx_vssrarni_h_w(a: v8i16, b: v8i16, c: u32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrarni.w.d"]
    fn __lsx_vssrarni_w_d(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrarni.d.q"]
    fn __lsx_vssrarni_d_q(a: v2i64, b: v2i64, c: u32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vssrarni.bu.h"]
    fn __lsx_vssrarni_bu_h(a: v16u8, b: v16i8, c: u32) -> v16u8;
    #[link_name = "llvm.loongarch.lsx.vssrarni.hu.w"]
    fn __lsx_vssrarni_hu_w(a: v8u16, b: v8i16, c: u32) -> v8u16;
    #[link_name = "llvm.loongarch.lsx.vssrarni.wu.d"]
    fn __lsx_vssrarni_wu_d(a: v4u32, b: v4i32, c: u32) -> v4u32;
    #[link_name = "llvm.loongarch.lsx.vssrarni.du.q"]
    fn __lsx_vssrarni_du_q(a: v2u64, b: v2i64, c: u32) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.vpermi.w"]
    fn __lsx_vpermi_w(a: v4i32, b: v4i32, c: u32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vld"]
    fn __lsx_vld(a: *const i8, b: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vst"]
    fn __lsx_vst(a: v16i8, b: *mut i8, c: i32);
    #[link_name = "llvm.loongarch.lsx.vssrlrn.b.h"]
    fn __lsx_vssrlrn_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrlrn.h.w"]
    fn __lsx_vssrlrn_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrlrn.w.d"]
    fn __lsx_vssrlrn_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vssrln.b.h"]
    fn __lsx_vssrln_b_h(a: v8i16, b: v8i16) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vssrln.h.w"]
    fn __lsx_vssrln_h_w(a: v4i32, b: v4i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vssrln.w.d"]
    fn __lsx_vssrln_w_d(a: v2i64, b: v2i64) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vorn.v"]
    fn __lsx_vorn_v(a: v16i8, b: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vldi"]
    fn __lsx_vldi(a: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vshuf.b"]
    fn __lsx_vshuf_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vldx"]
    fn __lsx_vldx(a: *const i8, b: i64) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vstx"]
    fn __lsx_vstx(a: v16i8, b: *mut i8, c: i64);
    #[link_name = "llvm.loongarch.lsx.vextl.qu.du"]
    fn __lsx_vextl_qu_du(a: v2u64) -> v2u64;
    #[link_name = "llvm.loongarch.lsx.bnz.b"]
    fn __lsx_bnz_b(a: v16u8) -> i32;
    #[link_name = "llvm.loongarch.lsx.bnz.d"]
    fn __lsx_bnz_d(a: v2u64) -> i32;
    #[link_name = "llvm.loongarch.lsx.bnz.h"]
    fn __lsx_bnz_h(a: v8u16) -> i32;
    #[link_name = "llvm.loongarch.lsx.bnz.v"]
    fn __lsx_bnz_v(a: v16u8) -> i32;
    #[link_name = "llvm.loongarch.lsx.bnz.w"]
    fn __lsx_bnz_w(a: v4u32) -> i32;
    #[link_name = "llvm.loongarch.lsx.bz.b"]
    fn __lsx_bz_b(a: v16u8) -> i32;
    #[link_name = "llvm.loongarch.lsx.bz.d"]
    fn __lsx_bz_d(a: v2u64) -> i32;
    #[link_name = "llvm.loongarch.lsx.bz.h"]
    fn __lsx_bz_h(a: v8u16) -> i32;
    #[link_name = "llvm.loongarch.lsx.bz.v"]
    fn __lsx_bz_v(a: v16u8) -> i32;
    #[link_name = "llvm.loongarch.lsx.bz.w"]
    fn __lsx_bz_w(a: v4u32) -> i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.caf.d"]
    fn __lsx_vfcmp_caf_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.caf.s"]
    fn __lsx_vfcmp_caf_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.ceq.d"]
    fn __lsx_vfcmp_ceq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.ceq.s"]
    fn __lsx_vfcmp_ceq_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cle.d"]
    fn __lsx_vfcmp_cle_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cle.s"]
    fn __lsx_vfcmp_cle_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.clt.d"]
    fn __lsx_vfcmp_clt_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.clt.s"]
    fn __lsx_vfcmp_clt_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cne.d"]
    fn __lsx_vfcmp_cne_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cne.s"]
    fn __lsx_vfcmp_cne_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cor.d"]
    fn __lsx_vfcmp_cor_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cor.s"]
    fn __lsx_vfcmp_cor_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cueq.d"]
    fn __lsx_vfcmp_cueq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cueq.s"]
    fn __lsx_vfcmp_cueq_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cule.d"]
    fn __lsx_vfcmp_cule_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cule.s"]
    fn __lsx_vfcmp_cule_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cult.d"]
    fn __lsx_vfcmp_cult_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cult.s"]
    fn __lsx_vfcmp_cult_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cun.d"]
    fn __lsx_vfcmp_cun_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cune.d"]
    fn __lsx_vfcmp_cune_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cune.s"]
    fn __lsx_vfcmp_cune_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.cun.s"]
    fn __lsx_vfcmp_cun_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.saf.d"]
    fn __lsx_vfcmp_saf_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.saf.s"]
    fn __lsx_vfcmp_saf_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.seq.d"]
    fn __lsx_vfcmp_seq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.seq.s"]
    fn __lsx_vfcmp_seq_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sle.d"]
    fn __lsx_vfcmp_sle_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sle.s"]
    fn __lsx_vfcmp_sle_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.slt.d"]
    fn __lsx_vfcmp_slt_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.slt.s"]
    fn __lsx_vfcmp_slt_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sne.d"]
    fn __lsx_vfcmp_sne_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sne.s"]
    fn __lsx_vfcmp_sne_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sor.d"]
    fn __lsx_vfcmp_sor_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sor.s"]
    fn __lsx_vfcmp_sor_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sueq.d"]
    fn __lsx_vfcmp_sueq_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sueq.s"]
    fn __lsx_vfcmp_sueq_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sule.d"]
    fn __lsx_vfcmp_sule_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sule.s"]
    fn __lsx_vfcmp_sule_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sult.d"]
    fn __lsx_vfcmp_sult_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sult.s"]
    fn __lsx_vfcmp_sult_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sun.d"]
    fn __lsx_vfcmp_sun_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sune.d"]
    fn __lsx_vfcmp_sune_d(a: v2f64, b: v2f64) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sune.s"]
    fn __lsx_vfcmp_sune_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vfcmp.sun.s"]
    fn __lsx_vfcmp_sun_s(a: v4f32, b: v4f32) -> v4i32;
    #[link_name = "llvm.loongarch.lsx.vrepli.b"]
    fn __lsx_vrepli_b(a: i32) -> v16i8;
    #[link_name = "llvm.loongarch.lsx.vrepli.d"]
    fn __lsx_vrepli_d(a: i32) -> v2i64;
    #[link_name = "llvm.loongarch.lsx.vrepli.h"]
    fn __lsx_vrepli_h(a: i32) -> v8i16;
    #[link_name = "llvm.loongarch.lsx.vrepli.w"]
    fn __lsx_vrepli_w(a: i32) -> v4i32;
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsll_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsll_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsll_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsll_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsll_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsll_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsll_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsll_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslli_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vslli_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslli_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vslli_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslli_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslli_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslli_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vslli_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsra_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsra_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsra_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsra_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsra_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsra_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsra_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsra_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrai_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsrai_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrai_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrai_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrai_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrai_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrai_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrai_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrar_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsrar_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrar_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsrar_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrar_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsrar_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrar_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsrar_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrari_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsrari_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrari_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrari_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrari_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrari_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrari_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrari_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrl_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsrl_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrl_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsrl_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrl_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsrl_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrl_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsrl_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrli_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsrli_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrli_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrli_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrli_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrli_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrli_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrli_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlr_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsrlr_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlr_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsrlr_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlr_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsrlr_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlr_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsrlr_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlri_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsrlri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlri_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrlri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlri_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrlri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlri_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrlri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclr_b(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vbitclr_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclr_h(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vbitclr_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclr_w(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vbitclr_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclr_d(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vbitclr_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclri_b<const IMM3: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vbitclri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclri_h<const IMM4: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vbitclri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclri_w<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vbitclri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitclri_d<const IMM6: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vbitclri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitset_b(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vbitset_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitset_h(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vbitset_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitset_w(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vbitset_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitset_d(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vbitset_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitseti_b<const IMM3: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vbitseti_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitseti_h<const IMM4: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vbitseti_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitseti_w<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vbitseti_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitseti_d<const IMM6: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vbitseti_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrev_b(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vbitrev_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrev_h(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vbitrev_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrev_w(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vbitrev_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrev_d(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vbitrev_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrevi_b<const IMM3: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vbitrevi_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrevi_h<const IMM4: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vbitrevi_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrevi_w<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vbitrevi_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitrevi_d<const IMM6: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vbitrevi_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadd_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vadd_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadd_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vadd_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadd_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vadd_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadd_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddi_bu<const IMM5: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vaddi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddi_hu<const IMM5: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vaddi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddi_wu<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vaddi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddi_du<const IMM5: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vaddi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsub_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsub_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsub_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsub_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsub_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsub_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsub_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsub_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubi_bu<const IMM5: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsubi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubi_hu<const IMM5: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsubi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubi_wu<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsubi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubi_du<const IMM5: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsubi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vmax_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vmax_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vmax_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmax_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmaxi_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmaxi_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmaxi_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmaxi_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vmax_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vmax_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vmax_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmax_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vmax_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_bu<const IMM5: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmaxi_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_hu<const IMM5: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmaxi_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_wu<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmaxi_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaxi_du<const IMM5: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmaxi_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vmin_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vmin_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vmin_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmin_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmini_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmini_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmini_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vmini_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vmin_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vmin_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vmin_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmin_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vmin_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_bu<const IMM5: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmini_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_hu<const IMM5: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmini_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_wu<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmini_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmini_du<const IMM5: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vmini_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseq_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vseq_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseq_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vseq_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseq_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vseq_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseq_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vseq_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseqi_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vseqi_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseqi_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vseqi_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseqi_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vseqi_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vseqi_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vseqi_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslti_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vslt_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vslt_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vslt_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vslt_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslti_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslti_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslti_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_bu(a: v16u8, b: v16u8) -> v16i8 {
    __lsx_vslt_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_hu(a: v8u16, b: v8u16) -> v8i16 {
    __lsx_vslt_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_wu(a: v4u32, b: v4u32) -> v4i32 {
    __lsx_vslt_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslt_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vslt_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_bu<const IMM5: u32>(a: v16u8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslti_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_hu<const IMM5: u32>(a: v8u16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslti_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_wu<const IMM5: u32>(a: v4u32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslti_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslti_du<const IMM5: u32>(a: v2u64) -> v2i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslti_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsle_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsle_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsle_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsle_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_b<const IMM_S5: i32>(a: v16i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslei_b(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_h<const IMM_S5: i32>(a: v8i16) -> v8i16 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslei_h(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_w<const IMM_S5: i32>(a: v4i32) -> v4i32 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslei_w(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_d<const IMM_S5: i32>(a: v2i64) -> v2i64 {
    static_assert_simm_bits!(IMM_S5, 5);
    __lsx_vslei_d(a, IMM_S5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_bu(a: v16u8, b: v16u8) -> v16i8 {
    __lsx_vsle_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_hu(a: v8u16, b: v8u16) -> v8i16 {
    __lsx_vsle_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_wu(a: v4u32, b: v4u32) -> v4i32 {
    __lsx_vsle_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsle_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vsle_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_bu<const IMM5: u32>(a: v16u8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslei_bu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_hu<const IMM5: u32>(a: v8u16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslei_hu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_wu<const IMM5: u32>(a: v4u32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslei_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vslei_du<const IMM5: u32>(a: v2u64) -> v2i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vslei_du(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsat_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsat_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsat_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsat_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_bu<const IMM3: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsat_bu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_hu<const IMM4: u32>(a: v8u16) -> v8u16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsat_hu(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_wu<const IMM5: u32>(a: v4u32) -> v4u32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsat_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsat_du<const IMM6: u32>(a: v2u64) -> v2u64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsat_du(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadda_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vadda_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadda_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vadda_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadda_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vadda_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadda_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vadda_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsadd_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsadd_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsadd_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vsadd_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vsadd_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vsadd_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsadd_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vsadd_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vavg_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vavg_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vavg_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vavg_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vavg_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vavg_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vavg_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavg_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vavg_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vavgr_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vavgr_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vavgr_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vavgr_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vavgr_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vavgr_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vavgr_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vavgr_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vavgr_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vssub_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vssub_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vssub_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vssub_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vssub_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vssub_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vssub_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssub_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vssub_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vabsd_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vabsd_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vabsd_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vabsd_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vabsd_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vabsd_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vabsd_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vabsd_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vabsd_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmul_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vmul_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmul_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vmul_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmul_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vmul_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmul_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmul_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmadd_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    __lsx_vmadd_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmadd_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    __lsx_vmadd_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmadd_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    __lsx_vmadd_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmadd_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    __lsx_vmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmsub_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    __lsx_vmsub_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmsub_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    __lsx_vmsub_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmsub_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    __lsx_vmsub_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmsub_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    __lsx_vmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vdiv_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vdiv_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vdiv_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vdiv_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vdiv_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vdiv_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vdiv_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vdiv_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vdiv_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vhaddw_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vhaddw_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vhaddw_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_hu_bu(a: v16u8, b: v16u8) -> v8u16 {
    __lsx_vhaddw_hu_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_wu_hu(a: v8u16, b: v8u16) -> v4u32 {
    __lsx_vhaddw_wu_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_du_wu(a: v4u32, b: v4u32) -> v2u64 {
    __lsx_vhaddw_du_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vhsubw_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vhsubw_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vhsubw_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_hu_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vhsubw_hu_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_wu_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vhsubw_wu_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_du_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vhsubw_du_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vmod_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vmod_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vmod_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmod_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vmod_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vmod_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vmod_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmod_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vmod_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplve_b(a: v16i8, b: i32) -> v16i8 {
    __lsx_vreplve_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplve_h(a: v8i16, b: i32) -> v8i16 {
    __lsx_vreplve_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplve_w(a: v4i32, b: i32) -> v4i32 {
    __lsx_vreplve_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplve_d(a: v2i64, b: i32) -> v2i64 {
    __lsx_vreplve_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplvei_b<const IMM4: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vreplvei_b(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplvei_h<const IMM3: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vreplvei_h(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplvei_w<const IMM2: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM2, 2);
    __lsx_vreplvei_w(a, IMM2)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplvei_d<const IMM1: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM1, 1);
    __lsx_vreplvei_d(a, IMM1)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickev_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vpickev_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickev_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vpickev_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickev_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vpickev_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickev_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vpickev_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickod_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vpickod_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickod_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vpickod_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickod_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vpickod_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickod_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vpickod_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvh_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vilvh_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvh_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vilvh_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvh_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vilvh_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvh_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vilvh_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvl_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vilvl_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvl_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vilvl_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvl_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vilvl_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vilvl_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vilvl_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackev_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vpackev_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackev_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vpackev_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackev_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vpackev_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackev_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vpackev_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackod_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vpackod_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackod_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vpackod_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackod_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vpackod_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpackod_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vpackod_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    __lsx_vshuf_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf_w(a: v4i32, b: v4i32, c: v4i32) -> v4i32 {
    __lsx_vshuf_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    __lsx_vshuf_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vand_v(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vand_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vandi_b<const IMM8: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vandi_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vor_v(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vor_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vori_b<const IMM8: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vnor_v(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vnor_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vnori_b<const IMM8: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vnori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vxor_v(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vxor_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vxori_b<const IMM8: u32>(a: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vxori_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitsel_v(a: v16u8, b: v16u8, c: v16u8) -> v16u8 {
    __lsx_vbitsel_v(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbitseli_b<const IMM8: u32>(a: v16u8, b: v16u8) -> v16u8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vbitseli_b(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf4i_b<const IMM8: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vshuf4i_b(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf4i_h<const IMM8: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vshuf4i_h(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf4i_w<const IMM8: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vshuf4i_w(a, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplgr2vr_b(a: i32) -> v16i8 {
    __lsx_vreplgr2vr_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplgr2vr_h(a: i32) -> v8i16 {
    __lsx_vreplgr2vr_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplgr2vr_w(a: i32) -> v4i32 {
    __lsx_vreplgr2vr_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vreplgr2vr_d(a: i64) -> v2i64 {
    __lsx_vreplgr2vr_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpcnt_b(a: v16i8) -> v16i8 {
    __lsx_vpcnt_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpcnt_h(a: v8i16) -> v8i16 {
    __lsx_vpcnt_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpcnt_w(a: v4i32) -> v4i32 {
    __lsx_vpcnt_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpcnt_d(a: v2i64) -> v2i64 {
    __lsx_vpcnt_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclo_b(a: v16i8) -> v16i8 {
    __lsx_vclo_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclo_h(a: v8i16) -> v8i16 {
    __lsx_vclo_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclo_w(a: v4i32) -> v4i32 {
    __lsx_vclo_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclo_d(a: v2i64) -> v2i64 {
    __lsx_vclo_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclz_b(a: v16i8) -> v16i8 {
    __lsx_vclz_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclz_h(a: v8i16) -> v8i16 {
    __lsx_vclz_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclz_w(a: v4i32) -> v4i32 {
    __lsx_vclz_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vclz_d(a: v2i64) -> v2i64 {
    __lsx_vclz_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_b<const IMM4: u32>(a: v16i8) -> i32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vpickve2gr_b(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_h<const IMM3: u32>(a: v8i16) -> i32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vpickve2gr_h(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_w<const IMM2: u32>(a: v4i32) -> i32 {
    static_assert_uimm_bits!(IMM2, 2);
    __lsx_vpickve2gr_w(a, IMM2)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_d<const IMM1: u32>(a: v2i64) -> i64 {
    static_assert_uimm_bits!(IMM1, 1);
    __lsx_vpickve2gr_d(a, IMM1)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_bu<const IMM4: u32>(a: v16i8) -> u32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vpickve2gr_bu(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_hu<const IMM3: u32>(a: v8i16) -> u32 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vpickve2gr_hu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_wu<const IMM2: u32>(a: v4i32) -> u32 {
    static_assert_uimm_bits!(IMM2, 2);
    __lsx_vpickve2gr_wu(a, IMM2)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpickve2gr_du<const IMM1: u32>(a: v2i64) -> u64 {
    static_assert_uimm_bits!(IMM1, 1);
    __lsx_vpickve2gr_du(a, IMM1)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vinsgr2vr_b<const IMM4: u32>(a: v16i8, b: i32) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vinsgr2vr_b(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vinsgr2vr_h<const IMM3: u32>(a: v8i16, b: i32) -> v8i16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vinsgr2vr_h(a, b, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vinsgr2vr_w<const IMM2: u32>(a: v4i32, b: i32) -> v4i32 {
    static_assert_uimm_bits!(IMM2, 2);
    __lsx_vinsgr2vr_w(a, b, IMM2)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vinsgr2vr_d<const IMM1: u32>(a: v2i64, b: i64) -> v2i64 {
    static_assert_uimm_bits!(IMM1, 1);
    __lsx_vinsgr2vr_d(a, b, IMM1)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfadd_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfadd_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfadd_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfadd_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfsub_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfsub_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfsub_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfsub_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmul_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfmul_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmul_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfmul_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfdiv_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfdiv_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfdiv_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfdiv_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvt_h_s(a: v4f32, b: v4f32) -> v8i16 {
    __lsx_vfcvt_h_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvt_s_d(a: v2f64, b: v2f64) -> v4f32 {
    __lsx_vfcvt_s_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmin_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfmin_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmin_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfmin_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmina_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfmina_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmina_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfmina_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmax_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfmax_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmax_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfmax_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmaxa_s(a: v4f32, b: v4f32) -> v4f32 {
    __lsx_vfmaxa_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmaxa_d(a: v2f64, b: v2f64) -> v2f64 {
    __lsx_vfmaxa_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfclass_s(a: v4f32) -> v4i32 {
    __lsx_vfclass_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfclass_d(a: v2f64) -> v2i64 {
    __lsx_vfclass_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfsqrt_s(a: v4f32) -> v4f32 {
    __lsx_vfsqrt_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfsqrt_d(a: v2f64) -> v2f64 {
    __lsx_vfsqrt_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrecip_s(a: v4f32) -> v4f32 {
    __lsx_vfrecip_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrecip_d(a: v2f64) -> v2f64 {
    __lsx_vfrecip_d(a)
}

#[inline]
#[target_feature(enable = "lsx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrecipe_s(a: v4f32) -> v4f32 {
    __lsx_vfrecipe_s(a)
}

#[inline]
#[target_feature(enable = "lsx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrecipe_d(a: v2f64) -> v2f64 {
    __lsx_vfrecipe_d(a)
}

#[inline]
#[target_feature(enable = "lsx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrsqrte_s(a: v4f32) -> v4f32 {
    __lsx_vfrsqrte_s(a)
}

#[inline]
#[target_feature(enable = "lsx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrsqrte_d(a: v2f64) -> v2f64 {
    __lsx_vfrsqrte_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrint_s(a: v4f32) -> v4f32 {
    __lsx_vfrint_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrint_d(a: v2f64) -> v2f64 {
    __lsx_vfrint_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrsqrt_s(a: v4f32) -> v4f32 {
    __lsx_vfrsqrt_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrsqrt_d(a: v2f64) -> v2f64 {
    __lsx_vfrsqrt_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vflogb_s(a: v4f32) -> v4f32 {
    __lsx_vflogb_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vflogb_d(a: v2f64) -> v2f64 {
    __lsx_vflogb_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvth_s_h(a: v8i16) -> v4f32 {
    __lsx_vfcvth_s_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvth_d_s(a: v4f32) -> v2f64 {
    __lsx_vfcvth_d_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvtl_s_h(a: v8i16) -> v4f32 {
    __lsx_vfcvtl_s_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcvtl_d_s(a: v4f32) -> v2f64 {
    __lsx_vfcvtl_d_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftint_w_s(a: v4f32) -> v4i32 {
    __lsx_vftint_w_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftint_l_d(a: v2f64) -> v2i64 {
    __lsx_vftint_l_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftint_wu_s(a: v4f32) -> v4u32 {
    __lsx_vftint_wu_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftint_lu_d(a: v2f64) -> v2u64 {
    __lsx_vftint_lu_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrz_w_s(a: v4f32) -> v4i32 {
    __lsx_vftintrz_w_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrz_l_d(a: v2f64) -> v2i64 {
    __lsx_vftintrz_l_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrz_wu_s(a: v4f32) -> v4u32 {
    __lsx_vftintrz_wu_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrz_lu_d(a: v2f64) -> v2u64 {
    __lsx_vftintrz_lu_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffint_s_w(a: v4i32) -> v4f32 {
    __lsx_vffint_s_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffint_d_l(a: v2i64) -> v2f64 {
    __lsx_vffint_d_l(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffint_s_wu(a: v4u32) -> v4f32 {
    __lsx_vffint_s_wu(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffint_d_lu(a: v2u64) -> v2f64 {
    __lsx_vffint_d_lu(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vandn_v(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vandn_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vneg_b(a: v16i8) -> v16i8 {
    __lsx_vneg_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vneg_h(a: v8i16) -> v8i16 {
    __lsx_vneg_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vneg_w(a: v4i32) -> v4i32 {
    __lsx_vneg_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vneg_d(a: v2i64) -> v2i64 {
    __lsx_vneg_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vmuh_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vmuh_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vmuh_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmuh_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_bu(a: v16u8, b: v16u8) -> v16u8 {
    __lsx_vmuh_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_hu(a: v8u16, b: v8u16) -> v8u16 {
    __lsx_vmuh_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_wu(a: v4u32, b: v4u32) -> v4u32 {
    __lsx_vmuh_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmuh_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vmuh_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_h_b<const IMM3: u32>(a: v16i8) -> v8i16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsllwil_h_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_w_h<const IMM4: u32>(a: v8i16) -> v4i32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsllwil_w_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_d_w<const IMM5: u32>(a: v4i32) -> v2i64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsllwil_d_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_hu_bu<const IMM3: u32>(a: v16u8) -> v8u16 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vsllwil_hu_bu(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_wu_hu<const IMM4: u32>(a: v8u16) -> v4u32 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsllwil_wu_hu(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsllwil_du_wu<const IMM5: u32>(a: v4u32) -> v2u64 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsllwil_du_wu(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsran_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vsran_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsran_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vsran_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsran_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vsran_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vssran_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vssran_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vssran_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_bu_h(a: v8u16, b: v8u16) -> v16u8 {
    __lsx_vssran_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_hu_w(a: v4u32, b: v4u32) -> v8u16 {
    __lsx_vssran_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssran_wu_d(a: v2u64, b: v2u64) -> v4u32 {
    __lsx_vssran_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarn_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vsrarn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarn_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vsrarn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarn_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vsrarn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vssrarn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vssrarn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vssrarn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_bu_h(a: v8u16, b: v8u16) -> v16u8 {
    __lsx_vssrarn_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_hu_w(a: v4u32, b: v4u32) -> v8u16 {
    __lsx_vssrarn_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarn_wu_d(a: v2u64, b: v2u64) -> v4u32 {
    __lsx_vssrarn_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrln_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vsrln_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrln_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vsrln_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrln_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vsrln_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_bu_h(a: v8u16, b: v8u16) -> v16u8 {
    __lsx_vssrln_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_hu_w(a: v4u32, b: v4u32) -> v8u16 {
    __lsx_vssrln_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_wu_d(a: v2u64, b: v2u64) -> v4u32 {
    __lsx_vssrln_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrn_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vsrlrn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrn_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vsrlrn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrn_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vsrlrn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_bu_h(a: v8u16, b: v8u16) -> v16u8 {
    __lsx_vssrlrn_bu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_hu_w(a: v4u32, b: v4u32) -> v8u16 {
    __lsx_vssrlrn_hu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_wu_d(a: v2u64, b: v2u64) -> v4u32 {
    __lsx_vssrlrn_wu_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrstpi_b<const IMM5: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vfrstpi_b(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrstpi_h<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vfrstpi_h(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrstp_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    __lsx_vfrstp_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrstp_h(a: v8i16, b: v8i16, c: v8i16) -> v8i16 {
    __lsx_vfrstp_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf4i_d<const IMM8: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vshuf4i_d(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbsrl_v<const IMM5: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vbsrl_v(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vbsll_v<const IMM5: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vbsll_v(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextrins_b<const IMM8: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vextrins_b(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextrins_h<const IMM8: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vextrins_h(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextrins_w<const IMM8: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vextrins_w(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextrins_d<const IMM8: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vextrins_d(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmskltz_b(a: v16i8) -> v16i8 {
    __lsx_vmskltz_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmskltz_h(a: v8i16) -> v8i16 {
    __lsx_vmskltz_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmskltz_w(a: v4i32) -> v4i32 {
    __lsx_vmskltz_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmskltz_d(a: v2i64) -> v2i64 {
    __lsx_vmskltz_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsigncov_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vsigncov_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsigncov_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vsigncov_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsigncov_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vsigncov_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsigncov_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsigncov_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmadd_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    __lsx_vfmadd_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    __lsx_vfmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmsub_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    __lsx_vfmsub_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    __lsx_vfmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfnmadd_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    __lsx_vfnmadd_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfnmadd_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    __lsx_vfnmadd_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfnmsub_s(a: v4f32, b: v4f32, c: v4f32) -> v4f32 {
    __lsx_vfnmsub_s(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfnmsub_d(a: v2f64, b: v2f64, c: v2f64) -> v2f64 {
    __lsx_vfnmsub_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrne_w_s(a: v4f32) -> v4i32 {
    __lsx_vftintrne_w_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrne_l_d(a: v2f64) -> v2i64 {
    __lsx_vftintrne_l_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrp_w_s(a: v4f32) -> v4i32 {
    __lsx_vftintrp_w_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrp_l_d(a: v2f64) -> v2i64 {
    __lsx_vftintrp_l_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrm_w_s(a: v4f32) -> v4i32 {
    __lsx_vftintrm_w_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrm_l_d(a: v2f64) -> v2i64 {
    __lsx_vftintrm_l_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftint_w_d(a: v2f64, b: v2f64) -> v4i32 {
    __lsx_vftint_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffint_s_l(a: v2i64, b: v2i64) -> v4f32 {
    __lsx_vffint_s_l(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrz_w_d(a: v2f64, b: v2f64) -> v4i32 {
    __lsx_vftintrz_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrp_w_d(a: v2f64, b: v2f64) -> v4i32 {
    __lsx_vftintrp_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrm_w_d(a: v2f64, b: v2f64) -> v4i32 {
    __lsx_vftintrm_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrne_w_d(a: v2f64, b: v2f64) -> v4i32 {
    __lsx_vftintrne_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintl_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintl_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftinth_l_s(a: v4f32) -> v2i64 {
    __lsx_vftinth_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffinth_d_w(a: v4i32) -> v2f64 {
    __lsx_vffinth_d_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vffintl_d_w(a: v4i32) -> v2f64 {
    __lsx_vffintl_d_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrzl_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrzl_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrzh_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrzh_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrpl_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrpl_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrph_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrph_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrml_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrml_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrmh_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrmh_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrnel_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrnel_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vftintrneh_l_s(a: v4f32) -> v2i64 {
    __lsx_vftintrneh_l_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrne_s(a: v4f32) -> v4f32 {
    __lsx_vfrintrne_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrne_d(a: v2f64) -> v2f64 {
    __lsx_vfrintrne_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrz_s(a: v4f32) -> v4f32 {
    __lsx_vfrintrz_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrz_d(a: v2f64) -> v2f64 {
    __lsx_vfrintrz_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrp_s(a: v4f32) -> v4f32 {
    __lsx_vfrintrp_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrp_d(a: v2f64) -> v2f64 {
    __lsx_vfrintrp_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrm_s(a: v4f32) -> v4f32 {
    __lsx_vfrintrm_s(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfrintrm_d(a: v2f64) -> v2f64 {
    __lsx_vfrintrm_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vstelm_b<const IMM_S8: i32, const IMM4: u32>(a: v16i8, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vstelm_b(a, mem_addr, IMM_S8, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vstelm_h<const IMM_S8: i32, const IMM3: u32>(a: v8i16, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vstelm_h(a, mem_addr, IMM_S8, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vstelm_w<const IMM_S8: i32, const IMM2: u32>(a: v4i32, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM2, 2);
    __lsx_vstelm_w(a, mem_addr, IMM_S8, IMM2)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vstelm_d<const IMM_S8: i32, const IMM1: u32>(a: v2i64, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM1, 1);
    __lsx_vstelm_d(a, mem_addr, IMM_S8, IMM1)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vaddwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vaddwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vaddwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vaddwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vaddwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vaddwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vaddwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vaddwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vaddwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vaddwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vaddwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vaddwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_d_wu_w(a: v4u32, b: v4i32) -> v2i64 {
    __lsx_vaddwev_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_w_hu_h(a: v8u16, b: v8i16) -> v4i32 {
    __lsx_vaddwev_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_h_bu_b(a: v16u8, b: v16i8) -> v8i16 {
    __lsx_vaddwev_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_d_wu_w(a: v4u32, b: v4i32) -> v2i64 {
    __lsx_vaddwod_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_w_hu_h(a: v8u16, b: v8i16) -> v4i32 {
    __lsx_vaddwod_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_h_bu_b(a: v16u8, b: v16i8) -> v8i16 {
    __lsx_vaddwod_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vsubwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vsubwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vsubwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vsubwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vsubwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vsubwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vsubwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vsubwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vsubwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vsubwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vsubwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vsubwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vaddwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vaddwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vaddwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vaddwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsubwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsubwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwev_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vsubwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsubwod_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vsubwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwev_q_du_d(a: v2u64, b: v2i64) -> v2i64 {
    __lsx_vaddwev_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vaddwod_q_du_d(a: v2u64, b: v2i64) -> v2i64 {
    __lsx_vaddwod_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vmulwev_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vmulwev_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vmulwev_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_d_w(a: v4i32, b: v4i32) -> v2i64 {
    __lsx_vmulwod_d_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_w_h(a: v8i16, b: v8i16) -> v4i32 {
    __lsx_vmulwod_w_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_h_b(a: v16i8, b: v16i8) -> v8i16 {
    __lsx_vmulwod_h_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vmulwev_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vmulwev_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vmulwev_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_d_wu(a: v4u32, b: v4u32) -> v2i64 {
    __lsx_vmulwod_d_wu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_w_hu(a: v8u16, b: v8u16) -> v4i32 {
    __lsx_vmulwod_w_hu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_h_bu(a: v16u8, b: v16u8) -> v8i16 {
    __lsx_vmulwod_h_bu(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_d_wu_w(a: v4u32, b: v4i32) -> v2i64 {
    __lsx_vmulwev_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_w_hu_h(a: v8u16, b: v8i16) -> v4i32 {
    __lsx_vmulwev_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_h_bu_b(a: v16u8, b: v16i8) -> v8i16 {
    __lsx_vmulwev_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_d_wu_w(a: v4u32, b: v4i32) -> v2i64 {
    __lsx_vmulwod_d_wu_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_w_hu_h(a: v8u16, b: v8i16) -> v4i32 {
    __lsx_vmulwod_w_hu_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_h_bu_b(a: v16u8, b: v16i8) -> v8i16 {
    __lsx_vmulwod_h_bu_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmulwev_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vmulwod_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vmulwev_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_q_du(a: v2u64, b: v2u64) -> v2i64 {
    __lsx_vmulwod_q_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwev_q_du_d(a: v2u64, b: v2i64) -> v2i64 {
    __lsx_vmulwev_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmulwod_q_du_d(a: v2u64, b: v2i64) -> v2i64 {
    __lsx_vmulwod_q_du_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vhaddw_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhaddw_qu_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vhaddw_qu_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_q_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vhsubw_q_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vhsubw_qu_du(a: v2u64, b: v2u64) -> v2u64 {
    __lsx_vhsubw_qu_du(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_d_w(a: v2i64, b: v4i32, c: v4i32) -> v2i64 {
    __lsx_vmaddwev_d_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_w_h(a: v4i32, b: v8i16, c: v8i16) -> v4i32 {
    __lsx_vmaddwev_w_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_h_b(a: v8i16, b: v16i8, c: v16i8) -> v8i16 {
    __lsx_vmaddwev_h_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_d_wu(a: v2u64, b: v4u32, c: v4u32) -> v2u64 {
    __lsx_vmaddwev_d_wu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_w_hu(a: v4u32, b: v8u16, c: v8u16) -> v4u32 {
    __lsx_vmaddwev_w_hu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_h_bu(a: v8u16, b: v16u8, c: v16u8) -> v8u16 {
    __lsx_vmaddwev_h_bu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_d_w(a: v2i64, b: v4i32, c: v4i32) -> v2i64 {
    __lsx_vmaddwod_d_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_w_h(a: v4i32, b: v8i16, c: v8i16) -> v4i32 {
    __lsx_vmaddwod_w_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_h_b(a: v8i16, b: v16i8, c: v16i8) -> v8i16 {
    __lsx_vmaddwod_h_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_d_wu(a: v2u64, b: v4u32, c: v4u32) -> v2u64 {
    __lsx_vmaddwod_d_wu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_w_hu(a: v4u32, b: v8u16, c: v8u16) -> v4u32 {
    __lsx_vmaddwod_w_hu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_h_bu(a: v8u16, b: v16u8, c: v16u8) -> v8u16 {
    __lsx_vmaddwod_h_bu(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_d_wu_w(a: v2i64, b: v4u32, c: v4i32) -> v2i64 {
    __lsx_vmaddwev_d_wu_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_w_hu_h(a: v4i32, b: v8u16, c: v8i16) -> v4i32 {
    __lsx_vmaddwev_w_hu_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_h_bu_b(a: v8i16, b: v16u8, c: v16i8) -> v8i16 {
    __lsx_vmaddwev_h_bu_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_d_wu_w(a: v2i64, b: v4u32, c: v4i32) -> v2i64 {
    __lsx_vmaddwod_d_wu_w(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_w_hu_h(a: v4i32, b: v8u16, c: v8i16) -> v4i32 {
    __lsx_vmaddwod_w_hu_h(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_h_bu_b(a: v8i16, b: v16u8, c: v16i8) -> v8i16 {
    __lsx_vmaddwod_h_bu_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_q_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    __lsx_vmaddwev_q_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_q_d(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    __lsx_vmaddwod_q_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_q_du(a: v2u64, b: v2u64, c: v2u64) -> v2u64 {
    __lsx_vmaddwev_q_du(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_q_du(a: v2u64, b: v2u64, c: v2u64) -> v2u64 {
    __lsx_vmaddwod_q_du(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwev_q_du_d(a: v2i64, b: v2u64, c: v2i64) -> v2i64 {
    __lsx_vmaddwev_q_du_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmaddwod_q_du_d(a: v2i64, b: v2u64, c: v2i64) -> v2i64 {
    __lsx_vmaddwod_q_du_d(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotr_b(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vrotr_b(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotr_h(a: v8i16, b: v8i16) -> v8i16 {
    __lsx_vrotr_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotr_w(a: v4i32, b: v4i32) -> v4i32 {
    __lsx_vrotr_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotr_d(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vrotr_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vadd_q(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vadd_q(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsub_q(a: v2i64, b: v2i64) -> v2i64 {
    __lsx_vsub_q(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldrepl_b<const IMM_S12: i32>(mem_addr: *const i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S12, 12);
    __lsx_vldrepl_b(mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldrepl_h<const IMM_S11: i32>(mem_addr: *const i8) -> v8i16 {
    static_assert_simm_bits!(IMM_S11, 11);
    __lsx_vldrepl_h(mem_addr, IMM_S11)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldrepl_w<const IMM_S10: i32>(mem_addr: *const i8) -> v4i32 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lsx_vldrepl_w(mem_addr, IMM_S10)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldrepl_d<const IMM_S9: i32>(mem_addr: *const i8) -> v2i64 {
    static_assert_simm_bits!(IMM_S9, 9);
    __lsx_vldrepl_d(mem_addr, IMM_S9)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmskgez_b(a: v16i8) -> v16i8 {
    __lsx_vmskgez_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vmsknz_b(a: v16i8) -> v16i8 {
    __lsx_vmsknz_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_h_b(a: v16i8) -> v8i16 {
    __lsx_vexth_h_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_w_h(a: v8i16) -> v4i32 {
    __lsx_vexth_w_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_d_w(a: v4i32) -> v2i64 {
    __lsx_vexth_d_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_q_d(a: v2i64) -> v2i64 {
    __lsx_vexth_q_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_hu_bu(a: v16u8) -> v8u16 {
    __lsx_vexth_hu_bu(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_wu_hu(a: v8u16) -> v4u32 {
    __lsx_vexth_wu_hu(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_du_wu(a: v4u32) -> v2u64 {
    __lsx_vexth_du_wu(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vexth_qu_du(a: v2u64) -> v2u64 {
    __lsx_vexth_qu_du(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotri_b<const IMM3: u32>(a: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM3, 3);
    __lsx_vrotri_b(a, IMM3)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotri_h<const IMM4: u32>(a: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vrotri_h(a, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotri_w<const IMM5: u32>(a: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vrotri_w(a, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrotri_d<const IMM6: u32>(a: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vrotri_d(a, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextl_q_d(a: v2i64) -> v2i64 {
    __lsx_vextl_q_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrlni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrlni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrlni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vsrlni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrlrni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrlrni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrlrni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrlrni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vsrlrni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrlni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrlni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrlni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrlni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_bu_h<const IMM4: u32>(a: v16u8, b: v16i8) -> v16u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrlni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_hu_w<const IMM5: u32>(a: v8u16, b: v8i16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrlni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_wu_d<const IMM6: u32>(a: v4u32, b: v4i32) -> v4u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrlni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlni_du_q<const IMM7: u32>(a: v2u64, b: v2i64) -> v2u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrlni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrlrni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrlrni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrlrni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrlrni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_bu_h<const IMM4: u32>(a: v16u8, b: v16i8) -> v16u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrlrni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_hu_w<const IMM5: u32>(a: v8u16, b: v8i16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrlrni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_wu_d<const IMM6: u32>(a: v4u32, b: v4i32) -> v4u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrlrni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrni_du_q<const IMM7: u32>(a: v2u64, b: v2i64) -> v2u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrlrni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrani_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrani_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrani_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrani_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrani_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrani_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrani_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vsrani_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vsrarni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vsrarni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vsrarni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vsrarni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vsrarni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrani_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrani_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrani_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrani_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_bu_h<const IMM4: u32>(a: v16u8, b: v16i8) -> v16u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrani_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_hu_w<const IMM5: u32>(a: v8u16, b: v8i16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrani_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_wu_d<const IMM6: u32>(a: v4u32, b: v4i32) -> v4u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrani_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrani_du_q<const IMM7: u32>(a: v2u64, b: v2i64) -> v2u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrani_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_b_h<const IMM4: u32>(a: v16i8, b: v16i8) -> v16i8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrarni_b_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_h_w<const IMM5: u32>(a: v8i16, b: v8i16) -> v8i16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrarni_h_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_w_d<const IMM6: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrarni_w_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_d_q<const IMM7: u32>(a: v2i64, b: v2i64) -> v2i64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrarni_d_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_bu_h<const IMM4: u32>(a: v16u8, b: v16i8) -> v16u8 {
    static_assert_uimm_bits!(IMM4, 4);
    __lsx_vssrarni_bu_h(a, b, IMM4)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_hu_w<const IMM5: u32>(a: v8u16, b: v8i16) -> v8u16 {
    static_assert_uimm_bits!(IMM5, 5);
    __lsx_vssrarni_hu_w(a, b, IMM5)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_wu_d<const IMM6: u32>(a: v4u32, b: v4i32) -> v4u32 {
    static_assert_uimm_bits!(IMM6, 6);
    __lsx_vssrarni_wu_d(a, b, IMM6)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrarni_du_q<const IMM7: u32>(a: v2u64, b: v2i64) -> v2u64 {
    static_assert_uimm_bits!(IMM7, 7);
    __lsx_vssrarni_du_q(a, b, IMM7)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vpermi_w<const IMM8: u32>(a: v4i32, b: v4i32) -> v4i32 {
    static_assert_uimm_bits!(IMM8, 8);
    __lsx_vpermi_w(a, b, IMM8)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vld<const IMM_S12: i32>(mem_addr: *const i8) -> v16i8 {
    static_assert_simm_bits!(IMM_S12, 12);
    __lsx_vld(mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vst<const IMM_S12: i32>(a: v16i8, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S12, 12);
    __lsx_vst(a, mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vssrlrn_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vssrlrn_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrlrn_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vssrlrn_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_b_h(a: v8i16, b: v8i16) -> v16i8 {
    __lsx_vssrln_b_h(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_h_w(a: v4i32, b: v4i32) -> v8i16 {
    __lsx_vssrln_h_w(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vssrln_w_d(a: v2i64, b: v2i64) -> v4i32 {
    __lsx_vssrln_w_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vorn_v(a: v16i8, b: v16i8) -> v16i8 {
    __lsx_vorn_v(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldi<const IMM_S13: i32>() -> v2i64 {
    static_assert_simm_bits!(IMM_S13, 13);
    __lsx_vldi(IMM_S13)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vshuf_b(a: v16i8, b: v16i8, c: v16i8) -> v16i8 {
    __lsx_vshuf_b(a, b, c)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vldx(mem_addr: *const i8, b: i64) -> v16i8 {
    __lsx_vldx(mem_addr, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vstx(a: v16i8, mem_addr: *mut i8, b: i64) {
    __lsx_vstx(a, mem_addr, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vextl_qu_du(a: v2u64) -> v2u64 {
    __lsx_vextl_qu_du(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bnz_b(a: v16u8) -> i32 {
    __lsx_bnz_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bnz_d(a: v2u64) -> i32 {
    __lsx_bnz_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bnz_h(a: v8u16) -> i32 {
    __lsx_bnz_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bnz_v(a: v16u8) -> i32 {
    __lsx_bnz_v(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bnz_w(a: v4u32) -> i32 {
    __lsx_bnz_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bz_b(a: v16u8) -> i32 {
    __lsx_bz_b(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bz_d(a: v2u64) -> i32 {
    __lsx_bz_d(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bz_h(a: v8u16) -> i32 {
    __lsx_bz_h(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bz_v(a: v16u8) -> i32 {
    __lsx_bz_v(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_bz_w(a: v4u32) -> i32 {
    __lsx_bz_w(a)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_caf_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_caf_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_caf_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_caf_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_ceq_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_ceq_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_ceq_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_ceq_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cle_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cle_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cle_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cle_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_clt_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_clt_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_clt_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_clt_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cne_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cne_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cne_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cne_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cor_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cor_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cor_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cor_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cueq_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cueq_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cueq_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cueq_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cule_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cule_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cule_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cule_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cult_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cult_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cult_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cult_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cun_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cun_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cune_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_cune_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cune_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cune_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_cun_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_cun_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_saf_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_saf_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_saf_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_saf_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_seq_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_seq_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_seq_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_seq_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sle_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sle_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sle_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sle_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_slt_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_slt_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_slt_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_slt_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sne_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sne_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sne_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sne_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sor_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sor_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sor_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sor_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sueq_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sueq_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sueq_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sueq_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sule_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sule_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sule_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sule_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sult_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sult_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sult_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sult_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sun_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sun_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sune_d(a: v2f64, b: v2f64) -> v2i64 {
    __lsx_vfcmp_sune_d(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sune_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sune_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vfcmp_sun_s(a: v4f32, b: v4f32) -> v4i32 {
    __lsx_vfcmp_sun_s(a, b)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrepli_b<const IMM_S10: i32>() -> v16i8 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lsx_vrepli_b(IMM_S10)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrepli_d<const IMM_S10: i32>() -> v2i64 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lsx_vrepli_d(IMM_S10)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrepli_h<const IMM_S10: i32>() -> v8i16 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lsx_vrepli_h(IMM_S10)
}

#[inline]
#[target_feature(enable = "lsx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lsx_vrepli_w<const IMM_S10: i32>() -> v4i32 {
    static_assert_simm_bits!(IMM_S10, 10);
    __lsx_vrepli_w(IMM_S10)
}
