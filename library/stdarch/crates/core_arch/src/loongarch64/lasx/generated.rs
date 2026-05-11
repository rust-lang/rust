// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-loongarch/lasx.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- crates/stdarch-gen-loongarch/lasx.spec
// ```

use crate::mem::transmute;
use super::super::*;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.lasx.xvsrar.b"]
    fn __lasx_xvsrar_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrar.h"]
    fn __lasx_xvsrar_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrar.w"]
    fn __lasx_xvsrar_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrar.d"]
    fn __lasx_xvsrar_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrari.b"]
    fn __lasx_xvsrari_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrari.h"]
    fn __lasx_xvsrari_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrari.w"]
    fn __lasx_xvsrari_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrari.d"]
    fn __lasx_xvsrari_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.b"]
    fn __lasx_xvsrlr_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.h"]
    fn __lasx_xvsrlr_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.w"]
    fn __lasx_xvsrlr_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlr.d"]
    fn __lasx_xvsrlr_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.b"]
    fn __lasx_xvsrlri_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.h"]
    fn __lasx_xvsrlri_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.w"]
    fn __lasx_xvsrlri_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlri.d"]
    fn __lasx_xvsrlri_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.b"]
    fn __lasx_xvbitclr_b(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.h"]
    fn __lasx_xvbitclr_h(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.w"]
    fn __lasx_xvbitclr_w(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitclr.d"]
    fn __lasx_xvbitclr_d(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.b"]
    fn __lasx_xvbitclri_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.h"]
    fn __lasx_xvbitclri_h(a: __v16u16, b: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.w"]
    fn __lasx_xvbitclri_w(a: __v8u32, b: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitclri.d"]
    fn __lasx_xvbitclri_d(a: __v4u64, b: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitset.b"]
    fn __lasx_xvbitset_b(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitset.h"]
    fn __lasx_xvbitset_h(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitset.w"]
    fn __lasx_xvbitset_w(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitset.d"]
    fn __lasx_xvbitset_d(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.b"]
    fn __lasx_xvbitseti_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.h"]
    fn __lasx_xvbitseti_h(a: __v16u16, b: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.w"]
    fn __lasx_xvbitseti_w(a: __v8u32, b: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitseti.d"]
    fn __lasx_xvbitseti_d(a: __v4u64, b: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.b"]
    fn __lasx_xvbitrev_b(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.h"]
    fn __lasx_xvbitrev_h(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.w"]
    fn __lasx_xvbitrev_w(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitrev.d"]
    fn __lasx_xvbitrev_d(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.b"]
    fn __lasx_xvbitrevi_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.h"]
    fn __lasx_xvbitrevi_h(a: __v16u16, b: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.w"]
    fn __lasx_xvbitrevi_w(a: __v8u32, b: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvbitrevi.d"]
    fn __lasx_xvbitrevi_d(a: __v4u64, b: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsubi.bu"]
    fn __lasx_xvsubi_bu(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsubi.hu"]
    fn __lasx_xvsubi_hu(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubi.wu"]
    fn __lasx_xvsubi_wu(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubi.du"]
    fn __lasx_xvsubi_du(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsat.b"]
    fn __lasx_xvsat_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsat.h"]
    fn __lasx_xvsat_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsat.w"]
    fn __lasx_xvsat_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsat.d"]
    fn __lasx_xvsat_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsat.bu"]
    fn __lasx_xvsat_bu(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvsat.hu"]
    fn __lasx_xvsat_hu(a: __v16u16, b: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsat.wu"]
    fn __lasx_xvsat_wu(a: __v8u32, b: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsat.du"]
    fn __lasx_xvsat_du(a: __v4u64, b: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvadda.b"]
    fn __lasx_xvadda_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvadda.h"]
    fn __lasx_xvadda_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvadda.w"]
    fn __lasx_xvadda_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvadda.d"]
    fn __lasx_xvadda_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsadd.b"]
    fn __lasx_xvsadd_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsadd.h"]
    fn __lasx_xvsadd_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsadd.w"]
    fn __lasx_xvsadd_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsadd.d"]
    fn __lasx_xvsadd_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsadd.bu"]
    fn __lasx_xvsadd_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvsadd.hu"]
    fn __lasx_xvsadd_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsadd.wu"]
    fn __lasx_xvsadd_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsadd.du"]
    fn __lasx_xvsadd_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvavg.b"]
    fn __lasx_xvavg_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvavg.h"]
    fn __lasx_xvavg_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvavg.w"]
    fn __lasx_xvavg_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvavg.d"]
    fn __lasx_xvavg_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvavg.bu"]
    fn __lasx_xvavg_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvavg.hu"]
    fn __lasx_xvavg_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvavg.wu"]
    fn __lasx_xvavg_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvavg.du"]
    fn __lasx_xvavg_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvavgr.b"]
    fn __lasx_xvavgr_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvavgr.h"]
    fn __lasx_xvavgr_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvavgr.w"]
    fn __lasx_xvavgr_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvavgr.d"]
    fn __lasx_xvavgr_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvavgr.bu"]
    fn __lasx_xvavgr_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvavgr.hu"]
    fn __lasx_xvavgr_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvavgr.wu"]
    fn __lasx_xvavgr_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvavgr.du"]
    fn __lasx_xvavgr_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssub.b"]
    fn __lasx_xvssub_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssub.h"]
    fn __lasx_xvssub_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssub.w"]
    fn __lasx_xvssub_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssub.d"]
    fn __lasx_xvssub_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssub.bu"]
    fn __lasx_xvssub_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssub.hu"]
    fn __lasx_xvssub_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssub.wu"]
    fn __lasx_xvssub_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssub.du"]
    fn __lasx_xvssub_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvabsd.b"]
    fn __lasx_xvabsd_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvabsd.h"]
    fn __lasx_xvabsd_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvabsd.w"]
    fn __lasx_xvabsd_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvabsd.d"]
    fn __lasx_xvabsd_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvabsd.bu"]
    fn __lasx_xvabsd_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvabsd.hu"]
    fn __lasx_xvabsd_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvabsd.wu"]
    fn __lasx_xvabsd_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvabsd.du"]
    fn __lasx_xvabsd_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.h.b"]
    fn __lasx_xvhaddw_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.w.h"]
    fn __lasx_xvhaddw_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.d.w"]
    fn __lasx_xvhaddw_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.hu.bu"]
    fn __lasx_xvhaddw_hu_bu(a: __v32u8, b: __v32u8) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.wu.hu"]
    fn __lasx_xvhaddw_wu_hu(a: __v16u16, b: __v16u16) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.du.wu"]
    fn __lasx_xvhaddw_du_wu(a: __v8u32, b: __v8u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.h.b"]
    fn __lasx_xvhsubw_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.w.h"]
    fn __lasx_xvhsubw_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.d.w"]
    fn __lasx_xvhsubw_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.hu.bu"]
    fn __lasx_xvhsubw_hu_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.wu.hu"]
    fn __lasx_xvhsubw_wu_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.du.wu"]
    fn __lasx_xvhsubw_du_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.b"]
    fn __lasx_xvrepl128vei_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.h"]
    fn __lasx_xvrepl128vei_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.w"]
    fn __lasx_xvrepl128vei_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrepl128vei.d"]
    fn __lasx_xvrepl128vei_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickev.b"]
    fn __lasx_xvpickev_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpickev.h"]
    fn __lasx_xvpickev_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpickev.w"]
    fn __lasx_xvpickev_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickev.d"]
    fn __lasx_xvpickev_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickod.b"]
    fn __lasx_xvpickod_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpickod.h"]
    fn __lasx_xvpickod_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpickod.w"]
    fn __lasx_xvpickod_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickod.d"]
    fn __lasx_xvpickod_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvilvh.b"]
    fn __lasx_xvilvh_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvilvh.h"]
    fn __lasx_xvilvh_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvilvh.w"]
    fn __lasx_xvilvh_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvilvh.d"]
    fn __lasx_xvilvh_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvilvl.b"]
    fn __lasx_xvilvl_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvilvl.h"]
    fn __lasx_xvilvl_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvilvl.w"]
    fn __lasx_xvilvl_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvilvl.d"]
    fn __lasx_xvilvl_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpackev.b"]
    fn __lasx_xvpackev_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpackev.h"]
    fn __lasx_xvpackev_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpackev.w"]
    fn __lasx_xvpackev_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpackev.d"]
    fn __lasx_xvpackev_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpackod.b"]
    fn __lasx_xvpackod_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpackod.h"]
    fn __lasx_xvpackod_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvpackod.w"]
    fn __lasx_xvpackod_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpackod.d"]
    fn __lasx_xvpackod_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvshuf.b"]
    fn __lasx_xvshuf_b(a: __v32i8, b: __v32i8, c: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvshuf.h"]
    fn __lasx_xvshuf_h(a: __v16i16, b: __v16i16, c: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf.w"]
    fn __lasx_xvshuf_w(a: __v8i32, b: __v8i32, c: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvshuf.d"]
    fn __lasx_xvshuf_d(a: __v4i64, b: __v4i64, c: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvandi.b"]
    fn __lasx_xvandi_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvori.b"]
    fn __lasx_xvori_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvnori.b"]
    fn __lasx_xvnori_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvxori.b"]
    fn __lasx_xvxori_b(a: __v32u8, b: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitsel.v"]
    fn __lasx_xvbitsel_v(a: __v32u8, b: __v32u8, c: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvbitseli.b"]
    fn __lasx_xvbitseli_b(a: __v32u8, b: __v32u8, c: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.b"]
    fn __lasx_xvshuf4i_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.h"]
    fn __lasx_xvshuf4i_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.w"]
    fn __lasx_xvshuf4i_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvclo.b"]
    fn __lasx_xvclo_b(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvclo.h"]
    fn __lasx_xvclo_h(a: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvclo.w"]
    fn __lasx_xvclo_w(a: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvclo.d"]
    fn __lasx_xvclo_d(a: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcvt.h.s"]
    fn __lasx_xvfcvt_h_s(a: __v8f32, b: __v8f32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvfcvt.s.d"]
    fn __lasx_xvfcvt_s_d(a: __v4f64, b: __v4f64) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmin.s"]
    fn __lasx_xvfmin_s(a: __v8f32, b: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmin.d"]
    fn __lasx_xvfmin_d(a: __v4f64, b: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmina.s"]
    fn __lasx_xvfmina_s(a: __v8f32, b: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmina.d"]
    fn __lasx_xvfmina_d(a: __v4f64, b: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmax.s"]
    fn __lasx_xvfmax_s(a: __v8f32, b: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmax.d"]
    fn __lasx_xvfmax_d(a: __v4f64, b: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfmaxa.s"]
    fn __lasx_xvfmaxa_s(a: __v8f32, b: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfmaxa.d"]
    fn __lasx_xvfmaxa_d(a: __v4f64, b: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfclass.s"]
    fn __lasx_xvfclass_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfclass.d"]
    fn __lasx_xvfclass_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfrecip.s"]
    fn __lasx_xvfrecip_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrecip.d"]
    fn __lasx_xvfrecip_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrecipe.s"]
    fn __lasx_xvfrecipe_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrecipe.d"]
    fn __lasx_xvfrecipe_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrte.s"]
    fn __lasx_xvfrsqrte_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrte.d"]
    fn __lasx_xvfrsqrte_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrint.s"]
    fn __lasx_xvfrint_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrint.d"]
    fn __lasx_xvfrint_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrt.s"]
    fn __lasx_xvfrsqrt_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrsqrt.d"]
    fn __lasx_xvfrsqrt_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvflogb.s"]
    fn __lasx_xvflogb_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvflogb.d"]
    fn __lasx_xvflogb_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfcvth.s.h"]
    fn __lasx_xvfcvth_s_h(a: __v16i16) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfcvth.d.s"]
    fn __lasx_xvfcvth_d_s(a: __v8f32) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfcvtl.s.h"]
    fn __lasx_xvfcvtl_s_h(a: __v16i16) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfcvtl.d.s"]
    fn __lasx_xvfcvtl_d_s(a: __v8f32) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvftint.w.s"]
    fn __lasx_xvftint_w_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftint.l.d"]
    fn __lasx_xvftint_l_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftint.wu.s"]
    fn __lasx_xvftint_wu_s(a: __v8f32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvftint.lu.d"]
    fn __lasx_xvftint_lu_d(a: __v4f64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.w.s"]
    fn __lasx_xvftintrz_w_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.l.d"]
    fn __lasx_xvftintrz_l_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.wu.s"]
    fn __lasx_xvftintrz_wu_s(a: __v8f32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.lu.d"]
    fn __lasx_xvftintrz_lu_d(a: __v4f64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.w"]
    fn __lasx_xvffint_s_w(a: __v8i32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvffint.d.l"]
    fn __lasx_xvffint_d_l(a: __v4i64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.wu"]
    fn __lasx_xvffint_s_wu(a: __v8u32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvffint.d.lu"]
    fn __lasx_xvffint_d_lu(a: __v4u64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvreplve.b"]
    fn __lasx_xvreplve_b(a: __v32i8, b: i32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvreplve.h"]
    fn __lasx_xvreplve_h(a: __v16i16, b: i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvreplve.w"]
    fn __lasx_xvreplve_w(a: __v8i32, b: i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplve.d"]
    fn __lasx_xvreplve_d(a: __v4i64, b: i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpermi.w"]
    fn __lasx_xvpermi_w(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmuh.b"]
    fn __lasx_xvmuh_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmuh.h"]
    fn __lasx_xvmuh_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmuh.w"]
    fn __lasx_xvmuh_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmuh.d"]
    fn __lasx_xvmuh_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmuh.bu"]
    fn __lasx_xvmuh_bu(a: __v32u8, b: __v32u8) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvmuh.hu"]
    fn __lasx_xvmuh_hu(a: __v16u16, b: __v16u16) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmuh.wu"]
    fn __lasx_xvmuh_wu(a: __v8u32, b: __v8u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmuh.du"]
    fn __lasx_xvmuh_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.h.b"]
    fn __lasx_xvsllwil_h_b(a: __v32i8, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.w.h"]
    fn __lasx_xvsllwil_w_h(a: __v16i16, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.d.w"]
    fn __lasx_xvsllwil_d_w(a: __v8i32, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.hu.bu"]
    fn __lasx_xvsllwil_hu_bu(a: __v32u8, b: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.wu.hu"]
    fn __lasx_xvsllwil_wu_hu(a: __v16u16, b: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsllwil.du.wu"]
    fn __lasx_xvsllwil_du_wu(a: __v8u32, b: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsran.b.h"]
    fn __lasx_xvsran_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsran.h.w"]
    fn __lasx_xvsran_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsran.w.d"]
    fn __lasx_xvsran_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssran.b.h"]
    fn __lasx_xvssran_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssran.h.w"]
    fn __lasx_xvssran_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssran.w.d"]
    fn __lasx_xvssran_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssran.bu.h"]
    fn __lasx_xvssran_bu_h(a: __v16u16, b: __v16u16) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssran.hu.w"]
    fn __lasx_xvssran_hu_w(a: __v8u32, b: __v8u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssran.wu.d"]
    fn __lasx_xvssran_wu_d(a: __v4u64, b: __v4u64) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.b.h"]
    fn __lasx_xvsrarn_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.h.w"]
    fn __lasx_xvsrarn_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrarn.w.d"]
    fn __lasx_xvsrarn_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.b.h"]
    fn __lasx_xvssrarn_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.h.w"]
    fn __lasx_xvssrarn_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.w.d"]
    fn __lasx_xvssrarn_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.bu.h"]
    fn __lasx_xvssrarn_bu_h(a: __v16u16, b: __v16u16) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.hu.w"]
    fn __lasx_xvssrarn_hu_w(a: __v8u32, b: __v8u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrarn.wu.d"]
    fn __lasx_xvssrarn_wu_d(a: __v4u64, b: __v4u64) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrln.b.h"]
    fn __lasx_xvsrln_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrln.h.w"]
    fn __lasx_xvsrln_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrln.w.d"]
    fn __lasx_xvsrln_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrln.bu.h"]
    fn __lasx_xvssrln_bu_h(a: __v16u16, b: __v16u16) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrln.hu.w"]
    fn __lasx_xvssrln_hu_w(a: __v8u32, b: __v8u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrln.wu.d"]
    fn __lasx_xvssrln_wu_d(a: __v4u64, b: __v4u64) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.b.h"]
    fn __lasx_xvsrlrn_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.h.w"]
    fn __lasx_xvsrlrn_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlrn.w.d"]
    fn __lasx_xvsrlrn_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.bu.h"]
    fn __lasx_xvssrlrn_bu_h(a: __v16u16, b: __v16u16) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.hu.w"]
    fn __lasx_xvssrlrn_hu_w(a: __v8u32, b: __v8u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.wu.d"]
    fn __lasx_xvssrlrn_wu_d(a: __v4u64, b: __v4u64) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvfrstpi.b"]
    fn __lasx_xvfrstpi_b(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvfrstpi.h"]
    fn __lasx_xvfrstpi_h(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvfrstp.b"]
    fn __lasx_xvfrstp_b(a: __v32i8, b: __v32i8, c: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvfrstp.h"]
    fn __lasx_xvfrstp_h(a: __v16i16, b: __v16i16, c: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvshuf4i.d"]
    fn __lasx_xvshuf4i_d(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvbsrl.v"]
    fn __lasx_xvbsrl_v(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvbsll.v"]
    fn __lasx_xvbsll_v(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvextrins.b"]
    fn __lasx_xvextrins_b(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvextrins.h"]
    fn __lasx_xvextrins_h(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvextrins.w"]
    fn __lasx_xvextrins_w(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvextrins.d"]
    fn __lasx_xvextrins_d(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.b"]
    fn __lasx_xvmskltz_b(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.h"]
    fn __lasx_xvmskltz_h(a: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.w"]
    fn __lasx_xvmskltz_w(a: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmskltz.d"]
    fn __lasx_xvmskltz_d(a: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.b"]
    fn __lasx_xvsigncov_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.h"]
    fn __lasx_xvsigncov_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.w"]
    fn __lasx_xvsigncov_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsigncov.d"]
    fn __lasx_xvsigncov_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.w.s"]
    fn __lasx_xvftintrne_w_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.l.d"]
    fn __lasx_xvftintrne_l_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.w.s"]
    fn __lasx_xvftintrp_w_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.l.d"]
    fn __lasx_xvftintrp_l_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.w.s"]
    fn __lasx_xvftintrm_w_s(a: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.l.d"]
    fn __lasx_xvftintrm_l_d(a: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftint.w.d"]
    fn __lasx_xvftint_w_d(a: __v4f64, b: __v4f64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvffint.s.l"]
    fn __lasx_xvffint_s_l(a: __v4i64, b: __v4i64) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvftintrz.w.d"]
    fn __lasx_xvftintrz_w_d(a: __v4f64, b: __v4f64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrp.w.d"]
    fn __lasx_xvftintrp_w_d(a: __v4f64, b: __v4f64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrm.w.d"]
    fn __lasx_xvftintrm_w_d(a: __v4f64, b: __v4f64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftintrne.w.d"]
    fn __lasx_xvftintrne_w_d(a: __v4f64, b: __v4f64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvftinth.l.s"]
    fn __lasx_xvftinth_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintl.l.s"]
    fn __lasx_xvftintl_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvffinth.d.w"]
    fn __lasx_xvffinth_d_w(a: __v8i32) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvffintl.d.w"]
    fn __lasx_xvffintl_d_w(a: __v8i32) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvftintrzh.l.s"]
    fn __lasx_xvftintrzh_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrzl.l.s"]
    fn __lasx_xvftintrzl_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrph.l.s"]
    fn __lasx_xvftintrph_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrpl.l.s"]
    fn __lasx_xvftintrpl_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrmh.l.s"]
    fn __lasx_xvftintrmh_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrml.l.s"]
    fn __lasx_xvftintrml_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrneh.l.s"]
    fn __lasx_xvftintrneh_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvftintrnel.l.s"]
    fn __lasx_xvftintrnel_l_s(a: __v8f32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrne.s"]
    fn __lasx_xvfrintrne_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrne.d"]
    fn __lasx_xvfrintrne_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrz.s"]
    fn __lasx_xvfrintrz_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrz.d"]
    fn __lasx_xvfrintrz_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrp.s"]
    fn __lasx_xvfrintrp_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrp.d"]
    fn __lasx_xvfrintrp_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvfrintrm.s"]
    fn __lasx_xvfrintrm_s(a: __v8f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.xvfrintrm.d"]
    fn __lasx_xvfrintrm_d(a: __v4f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvld"]
    fn __lasx_xvld(a: *const i8, b: i32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvst"]
    fn __lasx_xvst(a: __v32i8, b: *mut i8, c: i32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.b"]
    fn __lasx_xvstelm_b(a: __v32i8, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.h"]
    fn __lasx_xvstelm_h(a: __v16i16, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.w"]
    fn __lasx_xvstelm_w(a: __v8i32, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvstelm.d"]
    fn __lasx_xvstelm_d(a: __v4i64, b: *mut i8, c: i32, d: u32);
    #[link_name = "llvm.loongarch.lasx.xvinsve0.w"]
    fn __lasx_xvinsve0_w(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvinsve0.d"]
    fn __lasx_xvinsve0_d(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpickve.w"]
    fn __lasx_xvpickve_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickve.d"]
    fn __lasx_xvpickve_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.b.h"]
    fn __lasx_xvssrlrn_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.h.w"]
    fn __lasx_xvssrlrn_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrn.w.d"]
    fn __lasx_xvssrlrn_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrln.b.h"]
    fn __lasx_xvssrln_b_h(a: __v16i16, b: __v16i16) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrln.h.w"]
    fn __lasx_xvssrln_h_w(a: __v8i32, b: __v8i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrln.w.d"]
    fn __lasx_xvssrln_w_d(a: __v4i64, b: __v4i64) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvldi"]
    fn __lasx_xvldi(a: i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvldx"]
    fn __lasx_xvldx(a: *const i8, b: i64) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvstx"]
    fn __lasx_xvstx(a: __v32i8, b: *mut i8, c: i64);
    #[link_name = "llvm.loongarch.lasx.xvextl.qu.du"]
    fn __lasx_xvextl_qu_du(a: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.b"]
    fn __lasx_xvreplve0_b(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.h"]
    fn __lasx_xvreplve0_h(a: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.w"]
    fn __lasx_xvreplve0_w(a: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.d"]
    fn __lasx_xvreplve0_d(a: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvreplve0.q"]
    fn __lasx_xvreplve0_q(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.vext2xv.h.b"]
    fn __lasx_vext2xv_h_b(a: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.vext2xv.w.h"]
    fn __lasx_vext2xv_w_h(a: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.w"]
    fn __lasx_vext2xv_d_w(a: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.w.b"]
    fn __lasx_vext2xv_w_b(a: __v32i8) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.h"]
    fn __lasx_vext2xv_d_h(a: __v16i16) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.d.b"]
    fn __lasx_vext2xv_d_b(a: __v32i8) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.hu.bu"]
    fn __lasx_vext2xv_hu_bu(a: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.vext2xv.wu.hu"]
    fn __lasx_vext2xv_wu_hu(a: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.wu"]
    fn __lasx_vext2xv_du_wu(a: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.wu.bu"]
    fn __lasx_vext2xv_wu_bu(a: __v32i8) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.hu"]
    fn __lasx_vext2xv_du_hu(a: __v16i16) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.vext2xv.du.bu"]
    fn __lasx_vext2xv_du_bu(a: __v32i8) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvpermi.q"]
    fn __lasx_xvpermi_q(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvpermi.d"]
    fn __lasx_xvpermi_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvperm.w"]
    fn __lasx_xvperm_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.b"]
    fn __lasx_xvldrepl_b(a: *const i8, b: i32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.h"]
    fn __lasx_xvldrepl_h(a: *const i8, b: i32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.w"]
    fn __lasx_xvldrepl_w(a: *const i8, b: i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvldrepl.d"]
    fn __lasx_xvldrepl_d(a: *const i8, b: i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.d"]
    fn __lasx_xvaddwev_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.w"]
    fn __lasx_xvaddwev_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.h"]
    fn __lasx_xvaddwev_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.b"]
    fn __lasx_xvaddwev_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.du"]
    fn __lasx_xvaddwev_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.wu"]
    fn __lasx_xvaddwev_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.hu"]
    fn __lasx_xvaddwev_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.bu"]
    fn __lasx_xvaddwev_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.q.d"]
    fn __lasx_xvsubwev_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.d.w"]
    fn __lasx_xvsubwev_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.w.h"]
    fn __lasx_xvsubwev_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.h.b"]
    fn __lasx_xvsubwev_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.q.du"]
    fn __lasx_xvsubwev_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.d.wu"]
    fn __lasx_xvsubwev_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.w.hu"]
    fn __lasx_xvsubwev_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwev.h.bu"]
    fn __lasx_xvsubwev_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.d"]
    fn __lasx_xvmulwev_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.w"]
    fn __lasx_xvmulwev_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.h"]
    fn __lasx_xvmulwev_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.b"]
    fn __lasx_xvmulwev_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.du"]
    fn __lasx_xvmulwev_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.wu"]
    fn __lasx_xvmulwev_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.hu"]
    fn __lasx_xvmulwev_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.bu"]
    fn __lasx_xvmulwev_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.d"]
    fn __lasx_xvaddwod_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.w"]
    fn __lasx_xvaddwod_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.h"]
    fn __lasx_xvaddwod_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.b"]
    fn __lasx_xvaddwod_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.du"]
    fn __lasx_xvaddwod_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.wu"]
    fn __lasx_xvaddwod_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.hu"]
    fn __lasx_xvaddwod_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.bu"]
    fn __lasx_xvaddwod_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.q.d"]
    fn __lasx_xvsubwod_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.d.w"]
    fn __lasx_xvsubwod_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.w.h"]
    fn __lasx_xvsubwod_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.h.b"]
    fn __lasx_xvsubwod_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.q.du"]
    fn __lasx_xvsubwod_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.d.wu"]
    fn __lasx_xvsubwod_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.w.hu"]
    fn __lasx_xvsubwod_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsubwod.h.bu"]
    fn __lasx_xvsubwod_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.d"]
    fn __lasx_xvmulwod_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.w"]
    fn __lasx_xvmulwod_d_w(a: __v8i32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.h"]
    fn __lasx_xvmulwod_w_h(a: __v16i16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.b"]
    fn __lasx_xvmulwod_h_b(a: __v32i8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.du"]
    fn __lasx_xvmulwod_q_du(a: __v4u64, b: __v4u64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.wu"]
    fn __lasx_xvmulwod_d_wu(a: __v8u32, b: __v8u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.hu"]
    fn __lasx_xvmulwod_w_hu(a: __v16u16, b: __v16u16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.bu"]
    fn __lasx_xvmulwod_h_bu(a: __v32u8, b: __v32u8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.d.wu.w"]
    fn __lasx_xvaddwev_d_wu_w(a: __v8u32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.w.hu.h"]
    fn __lasx_xvaddwev_w_hu_h(a: __v16u16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.h.bu.b"]
    fn __lasx_xvaddwev_h_bu_b(a: __v32u8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.d.wu.w"]
    fn __lasx_xvmulwev_d_wu_w(a: __v8u32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.w.hu.h"]
    fn __lasx_xvmulwev_w_hu_h(a: __v16u16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.h.bu.b"]
    fn __lasx_xvmulwev_h_bu_b(a: __v32u8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.d.wu.w"]
    fn __lasx_xvaddwod_d_wu_w(a: __v8u32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.w.hu.h"]
    fn __lasx_xvaddwod_w_hu_h(a: __v16u16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.h.bu.b"]
    fn __lasx_xvaddwod_h_bu_b(a: __v32u8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.d.wu.w"]
    fn __lasx_xvmulwod_d_wu_w(a: __v8u32, b: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.w.hu.h"]
    fn __lasx_xvmulwod_w_hu_h(a: __v16u16, b: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.h.bu.b"]
    fn __lasx_xvmulwod_h_bu_b(a: __v32u8, b: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.q.d"]
    fn __lasx_xvhaddw_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhaddw.qu.du"]
    fn __lasx_xvhaddw_qu_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.q.d"]
    fn __lasx_xvhsubw_q_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvhsubw.qu.du"]
    fn __lasx_xvhsubw_qu_du(a: __v4u64, b: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.d"]
    fn __lasx_xvmaddwev_q_d(a: __v4i64, b: __v4i64, c: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.w"]
    fn __lasx_xvmaddwev_d_w(a: __v4i64, b: __v8i32, c: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.h"]
    fn __lasx_xvmaddwev_w_h(a: __v8i32, b: __v16i16, c: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.b"]
    fn __lasx_xvmaddwev_h_b(a: __v16i16, b: __v32i8, c: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.du"]
    fn __lasx_xvmaddwev_q_du(a: __v4u64, b: __v4u64, c: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.wu"]
    fn __lasx_xvmaddwev_d_wu(a: __v4u64, b: __v8u32, c: __v8u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.hu"]
    fn __lasx_xvmaddwev_w_hu(a: __v8u32, b: __v16u16, c: __v16u16) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.bu"]
    fn __lasx_xvmaddwev_h_bu(a: __v16u16, b: __v32u8, c: __v32u8) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.d"]
    fn __lasx_xvmaddwod_q_d(a: __v4i64, b: __v4i64, c: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.w"]
    fn __lasx_xvmaddwod_d_w(a: __v4i64, b: __v8i32, c: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.h"]
    fn __lasx_xvmaddwod_w_h(a: __v8i32, b: __v16i16, c: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.b"]
    fn __lasx_xvmaddwod_h_b(a: __v16i16, b: __v32i8, c: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.du"]
    fn __lasx_xvmaddwod_q_du(a: __v4u64, b: __v4u64, c: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.wu"]
    fn __lasx_xvmaddwod_d_wu(a: __v4u64, b: __v8u32, c: __v8u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.hu"]
    fn __lasx_xvmaddwod_w_hu(a: __v8u32, b: __v16u16, c: __v16u16) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.bu"]
    fn __lasx_xvmaddwod_h_bu(a: __v16u16, b: __v32u8, c: __v32u8) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.q.du.d"]
    fn __lasx_xvmaddwev_q_du_d(a: __v4i64, b: __v4u64, c: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.d.wu.w"]
    fn __lasx_xvmaddwev_d_wu_w(a: __v4i64, b: __v8u32, c: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.w.hu.h"]
    fn __lasx_xvmaddwev_w_hu_h(a: __v8i32, b: __v16u16, c: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwev.h.bu.b"]
    fn __lasx_xvmaddwev_h_bu_b(a: __v16i16, b: __v32u8, c: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.q.du.d"]
    fn __lasx_xvmaddwod_q_du_d(a: __v4i64, b: __v4u64, c: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.d.wu.w"]
    fn __lasx_xvmaddwod_d_wu_w(a: __v4i64, b: __v8u32, c: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.w.hu.h"]
    fn __lasx_xvmaddwod_w_hu_h(a: __v8i32, b: __v16u16, c: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvmaddwod.h.bu.b"]
    fn __lasx_xvmaddwod_h_bu_b(a: __v16i16, b: __v32u8, c: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotr.b"]
    fn __lasx_xvrotr_b(a: __v32i8, b: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrotr.h"]
    fn __lasx_xvrotr_h(a: __v16i16, b: __v16i16) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotr.w"]
    fn __lasx_xvrotr_w(a: __v8i32, b: __v8i32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrotr.d"]
    fn __lasx_xvrotr_d(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvadd.q"]
    fn __lasx_xvadd_q(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsub.q"]
    fn __lasx_xvsub_q(a: __v4i64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwev.q.du.d"]
    fn __lasx_xvaddwev_q_du_d(a: __v4u64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvaddwod.q.du.d"]
    fn __lasx_xvaddwod_q_du_d(a: __v4u64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwev.q.du.d"]
    fn __lasx_xvmulwev_q_du_d(a: __v4u64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmulwod.q.du.d"]
    fn __lasx_xvmulwod_q_du_d(a: __v4u64, b: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvmskgez.b"]
    fn __lasx_xvmskgez_b(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvmsknz.b"]
    fn __lasx_xvmsknz_b(a: __v32i8) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvexth.h.b"]
    fn __lasx_xvexth_h_b(a: __v32i8) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvexth.w.h"]
    fn __lasx_xvexth_w_h(a: __v16i16) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvexth.d.w"]
    fn __lasx_xvexth_d_w(a: __v8i32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvexth.q.d"]
    fn __lasx_xvexth_q_d(a: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvexth.hu.bu"]
    fn __lasx_xvexth_hu_bu(a: __v32u8) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvexth.wu.hu"]
    fn __lasx_xvexth_wu_hu(a: __v16u16) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvexth.du.wu"]
    fn __lasx_xvexth_du_wu(a: __v8u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvexth.qu.du"]
    fn __lasx_xvexth_qu_du(a: __v4u64) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvrotri.b"]
    fn __lasx_xvrotri_b(a: __v32i8, b: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvrotri.h"]
    fn __lasx_xvrotri_h(a: __v16i16, b: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvrotri.w"]
    fn __lasx_xvrotri_w(a: __v8i32, b: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvrotri.d"]
    fn __lasx_xvrotri_d(a: __v4i64, b: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvextl.q.d"]
    fn __lasx_xvextl_q_d(a: __v4i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.b.h"]
    fn __lasx_xvsrlni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.h.w"]
    fn __lasx_xvsrlni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.w.d"]
    fn __lasx_xvsrlni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlni.d.q"]
    fn __lasx_xvsrlni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.b.h"]
    fn __lasx_xvsrlrni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.h.w"]
    fn __lasx_xvsrlrni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.w.d"]
    fn __lasx_xvsrlrni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrlrni.d.q"]
    fn __lasx_xvsrlrni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.b.h"]
    fn __lasx_xvssrlni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.h.w"]
    fn __lasx_xvssrlni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.w.d"]
    fn __lasx_xvssrlni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.d.q"]
    fn __lasx_xvssrlni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.bu.h"]
    fn __lasx_xvssrlni_bu_h(a: __v32u8, b: __v32i8, c: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.hu.w"]
    fn __lasx_xvssrlni_hu_w(a: __v16u16, b: __v16i16, c: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.wu.d"]
    fn __lasx_xvssrlni_wu_d(a: __v8u32, b: __v8i32, c: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrlni.du.q"]
    fn __lasx_xvssrlni_du_q(a: __v4u64, b: __v4i64, c: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.b.h"]
    fn __lasx_xvssrlrni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.h.w"]
    fn __lasx_xvssrlrni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.w.d"]
    fn __lasx_xvssrlrni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.d.q"]
    fn __lasx_xvssrlrni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.bu.h"]
    fn __lasx_xvssrlrni_bu_h(a: __v32u8, b: __v32i8, c: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.hu.w"]
    fn __lasx_xvssrlrni_hu_w(a: __v16u16, b: __v16i16, c: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.wu.d"]
    fn __lasx_xvssrlrni_wu_d(a: __v8u32, b: __v8i32, c: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrlrni.du.q"]
    fn __lasx_xvssrlrni_du_q(a: __v4u64, b: __v4i64, c: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvsrani.b.h"]
    fn __lasx_xvsrani_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrani.h.w"]
    fn __lasx_xvsrani_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrani.w.d"]
    fn __lasx_xvsrani_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrani.d.q"]
    fn __lasx_xvsrani_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.b.h"]
    fn __lasx_xvsrarni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.h.w"]
    fn __lasx_xvsrarni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.w.d"]
    fn __lasx_xvsrarni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvsrarni.d.q"]
    fn __lasx_xvsrarni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrani.b.h"]
    fn __lasx_xvssrani_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrani.h.w"]
    fn __lasx_xvssrani_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrani.w.d"]
    fn __lasx_xvssrani_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrani.d.q"]
    fn __lasx_xvssrani_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrani.bu.h"]
    fn __lasx_xvssrani_bu_h(a: __v32u8, b: __v32i8, c: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrani.hu.w"]
    fn __lasx_xvssrani_hu_w(a: __v16u16, b: __v16i16, c: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrani.wu.d"]
    fn __lasx_xvssrani_wu_d(a: __v8u32, b: __v8i32, c: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrani.du.q"]
    fn __lasx_xvssrani_du_q(a: __v4u64, b: __v4i64, c: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.b.h"]
    fn __lasx_xvssrarni_b_h(a: __v32i8, b: __v32i8, c: u32) -> __v32i8;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.h.w"]
    fn __lasx_xvssrarni_h_w(a: __v16i16, b: __v16i16, c: u32) -> __v16i16;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.w.d"]
    fn __lasx_xvssrarni_w_d(a: __v8i32, b: __v8i32, c: u32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.d.q"]
    fn __lasx_xvssrarni_d_q(a: __v4i64, b: __v4i64, c: u32) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.bu.h"]
    fn __lasx_xvssrarni_bu_h(a: __v32u8, b: __v32i8, c: u32) -> __v32u8;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.hu.w"]
    fn __lasx_xvssrarni_hu_w(a: __v16u16, b: __v16i16, c: u32) -> __v16u16;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.wu.d"]
    fn __lasx_xvssrarni_wu_d(a: __v8u32, b: __v8i32, c: u32) -> __v8u32;
    #[link_name = "llvm.loongarch.lasx.xvssrarni.du.q"]
    fn __lasx_xvssrarni_du_q(a: __v4u64, b: __v4i64, c: u32) -> __v4u64;
    #[link_name = "llvm.loongarch.lasx.xbnz.b"]
    fn __lasx_xbnz_b(a: __v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.d"]
    fn __lasx_xbnz_d(a: __v4u64) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.h"]
    fn __lasx_xbnz_h(a: __v16u16) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.v"]
    fn __lasx_xbnz_v(a: __v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbnz.w"]
    fn __lasx_xbnz_w(a: __v8u32) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.b"]
    fn __lasx_xbz_b(a: __v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.d"]
    fn __lasx_xbz_d(a: __v4u64) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.h"]
    fn __lasx_xbz_h(a: __v16u16) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.v"]
    fn __lasx_xbz_v(a: __v32u8) -> i32;
    #[link_name = "llvm.loongarch.lasx.xbz.w"]
    fn __lasx_xbz_w(a: __v8u32) -> i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.caf.d"]
    fn __lasx_xvfcmp_caf_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.caf.s"]
    fn __lasx_xvfcmp_caf_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.ceq.d"]
    fn __lasx_xvfcmp_ceq_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.ceq.s"]
    fn __lasx_xvfcmp_ceq_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cle.d"]
    fn __lasx_xvfcmp_cle_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cle.s"]
    fn __lasx_xvfcmp_cle_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.clt.d"]
    fn __lasx_xvfcmp_clt_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.clt.s"]
    fn __lasx_xvfcmp_clt_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cne.d"]
    fn __lasx_xvfcmp_cne_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cne.s"]
    fn __lasx_xvfcmp_cne_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cor.d"]
    fn __lasx_xvfcmp_cor_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cor.s"]
    fn __lasx_xvfcmp_cor_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cueq.d"]
    fn __lasx_xvfcmp_cueq_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cueq.s"]
    fn __lasx_xvfcmp_cueq_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cule.d"]
    fn __lasx_xvfcmp_cule_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cule.s"]
    fn __lasx_xvfcmp_cule_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cult.d"]
    fn __lasx_xvfcmp_cult_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cult.s"]
    fn __lasx_xvfcmp_cult_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cun.d"]
    fn __lasx_xvfcmp_cun_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cune.d"]
    fn __lasx_xvfcmp_cune_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cune.s"]
    fn __lasx_xvfcmp_cune_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.cun.s"]
    fn __lasx_xvfcmp_cun_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.saf.d"]
    fn __lasx_xvfcmp_saf_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.saf.s"]
    fn __lasx_xvfcmp_saf_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.seq.d"]
    fn __lasx_xvfcmp_seq_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.seq.s"]
    fn __lasx_xvfcmp_seq_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sle.d"]
    fn __lasx_xvfcmp_sle_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sle.s"]
    fn __lasx_xvfcmp_sle_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.slt.d"]
    fn __lasx_xvfcmp_slt_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.slt.s"]
    fn __lasx_xvfcmp_slt_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sne.d"]
    fn __lasx_xvfcmp_sne_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sne.s"]
    fn __lasx_xvfcmp_sne_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sor.d"]
    fn __lasx_xvfcmp_sor_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sor.s"]
    fn __lasx_xvfcmp_sor_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sueq.d"]
    fn __lasx_xvfcmp_sueq_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sueq.s"]
    fn __lasx_xvfcmp_sueq_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sule.d"]
    fn __lasx_xvfcmp_sule_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sule.s"]
    fn __lasx_xvfcmp_sule_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sult.d"]
    fn __lasx_xvfcmp_sult_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sult.s"]
    fn __lasx_xvfcmp_sult_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sun.d"]
    fn __lasx_xvfcmp_sun_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sune.d"]
    fn __lasx_xvfcmp_sune_d(a: __v4f64, b: __v4f64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sune.s"]
    fn __lasx_xvfcmp_sune_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvfcmp.sun.s"]
    fn __lasx_xvfcmp_sun_s(a: __v8f32, b: __v8f32) -> __v8i32;
    #[link_name = "llvm.loongarch.lasx.xvpickve.d.f"]
    fn __lasx_xvpickve_d_f(a: __v4f64, b: u32) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.xvpickve.w.f"]
    fn __lasx_xvpickve_w_f(a: __v8f32, b: u32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.cast.128.s"]
    fn __lasx_cast_128_s(a: __v4f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.cast.128.d"]
    fn __lasx_cast_128_d(a: __v2f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.cast.128"]
    fn __lasx_cast_128(a: __v2i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.concat.128.s"]
    fn __lasx_concat_128_s(a: __v4f32, b: __v4f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.concat.128.d"]
    fn __lasx_concat_128_d(a: __v2f64, b: __v2f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.concat.128"]
    fn __lasx_concat_128(a: __v2i64, b: __v2i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.extract.128.lo.s"]
    fn __lasx_extract_128_lo_s(a: __v8f32) -> __v4f32;
    #[link_name = "llvm.loongarch.lasx.extract.128.hi.s"]
    fn __lasx_extract_128_hi_s(a: __v8f32) -> __v4f32;
    #[link_name = "llvm.loongarch.lasx.extract.128.lo.d"]
    fn __lasx_extract_128_lo_d(a: __v4f64) -> __v2f64;
    #[link_name = "llvm.loongarch.lasx.extract.128.hi.d"]
    fn __lasx_extract_128_hi_d(a: __v4f64) -> __v2f64;
    #[link_name = "llvm.loongarch.lasx.extract.128.lo"]
    fn __lasx_extract_128_lo(a: __v4i64) -> __v2i64;
    #[link_name = "llvm.loongarch.lasx.extract.128.hi"]
    fn __lasx_extract_128_hi(a: __v4i64) -> __v2i64;
    #[link_name = "llvm.loongarch.lasx.insert.128.lo.s"]
    fn __lasx_insert_128_lo_s(a: __v8f32, b: __v4f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.insert.128.hi.s"]
    fn __lasx_insert_128_hi_s(a: __v8f32, b: __v4f32) -> __v8f32;
    #[link_name = "llvm.loongarch.lasx.insert.128.lo.d"]
    fn __lasx_insert_128_lo_d(a: __v4f64, b: __v2f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.insert.128.hi.d"]
    fn __lasx_insert_128_hi_d(a: __v4f64, b: __v2f64) -> __v4f64;
    #[link_name = "llvm.loongarch.lasx.insert.128.lo"]
    fn __lasx_insert_128_lo(a: __v4i64, b: __v2i64) -> __v4i64;
    #[link_name = "llvm.loongarch.lasx.insert.128.hi"]
    fn __lasx_insert_128_hi(a: __v4i64, b: __v2i64) -> __v4i64;
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrar_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrar_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrar_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrar_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrar_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrar_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrar_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrar_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrari_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsrari_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrari_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrari_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrari_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrari_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrari_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrari_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlr_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlr_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlr_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlr_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlr_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlr_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlr_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlr_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlri_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsrlri_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlri_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrlri_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlri_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrlri_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlri_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrlri_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclr_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitclr_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclr_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitclr_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclr_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitclr_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclr_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitclr_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclri_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvbitclri_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclri_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvbitclri_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclri_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvbitclri_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitclri_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvbitclri_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitset_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitset_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitset_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitset_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitset_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitset_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitset_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitset_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitseti_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvbitseti_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitseti_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvbitseti_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitseti_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvbitseti_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitseti_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvbitseti_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrev_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitrev_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrev_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitrev_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrev_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitrev_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrev_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitrev_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrevi_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvbitrevi_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrevi_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvbitrevi_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrevi_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvbitrevi_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitrevi_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvbitrevi_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubi_bu<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsubi_bu(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubi_hu<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsubi_hu(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubi_wu<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsubi_wu(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubi_du<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsubi_du(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsat_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsat_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsat_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsat_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_bu<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsat_bu(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_hu<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsat_hu(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_wu<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsat_wu(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsat_du<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsat_du(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvadda_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvadda_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvadda_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvadda_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvadda_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvadda_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvadda_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvadda_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsadd_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsadd_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavg_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavg_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvavgr_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvavgr_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssub_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssub_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvabsd_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvabsd_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_hu_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_hu_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_wu_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_wu_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_du_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_du_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_hu_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_hu_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_wu_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_wu_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_du_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_du_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrepl128vei_b<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvrepl128vei_b(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrepl128vei_h<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvrepl128vei_h(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrepl128vei_w<const IMM2: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM2, 2);
    unsafe { transmute(__lasx_xvrepl128vei_w(transmute(a), IMM2)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrepl128vei_d<const IMM1: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM1, 1);
    unsafe { transmute(__lasx_xvrepl128vei_d(transmute(a), IMM1)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickev_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickev_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickev_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickev_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickev_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickev_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickev_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickev_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickod_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickod_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickod_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickod_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickod_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickod_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickod_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpickod_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvh_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvh_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvh_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvh_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvh_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvh_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvh_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvh_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvl_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvl_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvl_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvl_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvl_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvl_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvilvl_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvilvl_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackev_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackev_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackev_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackev_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackev_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackev_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackev_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackev_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackod_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackod_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackod_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackod_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackod_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackod_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpackod_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvpackod_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvshuf_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvshuf_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf_w(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvshuf_w(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf_d(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvshuf_d(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvandi_b<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvandi_b(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvori_b<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvori_b(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvnori_b<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvnori_b(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvxori_b<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvxori_b(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitsel_v(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvbitsel_v(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbitseli_b<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvbitseli_b(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf4i_b<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvshuf4i_b(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf4i_h<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvshuf4i_h(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf4i_w<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvshuf4i_w(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvclo_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvclo_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvclo_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvclo_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvclo_w(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvclo_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvclo_d(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvclo_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvt_h_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcvt_h_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvt_s_d(a: m256d, b: m256d) -> m256 {
    unsafe { transmute(__lasx_xvfcvt_s_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmin_s(a: m256, b: m256) -> m256 {
    unsafe { transmute(__lasx_xvfmin_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmin_d(a: m256d, b: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfmin_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmina_s(a: m256, b: m256) -> m256 {
    unsafe { transmute(__lasx_xvfmina_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmina_d(a: m256d, b: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfmina_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmax_s(a: m256, b: m256) -> m256 {
    unsafe { transmute(__lasx_xvfmax_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmax_d(a: m256d, b: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfmax_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmaxa_s(a: m256, b: m256) -> m256 {
    unsafe { transmute(__lasx_xvfmaxa_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfmaxa_d(a: m256d, b: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfmaxa_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfclass_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvfclass_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfclass_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfclass_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrecip_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrecip_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrecip_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrecip_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrecipe_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrecipe_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrecipe_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrecipe_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrsqrte_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrsqrte_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx,frecipe")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrsqrte_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrsqrte_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrint_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrint_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrint_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrint_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrsqrt_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrsqrt_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrsqrt_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrsqrt_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvflogb_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvflogb_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvflogb_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvflogb_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvth_s_h(a: m256i) -> m256 {
    unsafe { transmute(__lasx_xvfcvth_s_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvth_d_s(a: m256) -> m256d {
    unsafe { transmute(__lasx_xvfcvth_d_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvtl_s_h(a: m256i) -> m256 {
    unsafe { transmute(__lasx_xvfcvtl_s_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcvtl_d_s(a: m256) -> m256d {
    unsafe { transmute(__lasx_xvfcvtl_d_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftint_w_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftint_w_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftint_l_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftint_l_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftint_wu_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftint_wu_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftint_lu_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftint_lu_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrz_w_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrz_w_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrz_l_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrz_l_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrz_wu_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrz_wu_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrz_lu_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrz_lu_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffint_s_w(a: m256i) -> m256 {
    unsafe { transmute(__lasx_xvffint_s_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffint_d_l(a: m256i) -> m256d {
    unsafe { transmute(__lasx_xvffint_d_l(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffint_s_wu(a: m256i) -> m256 {
    unsafe { transmute(__lasx_xvffint_s_wu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffint_d_lu(a: m256i) -> m256d {
    unsafe { transmute(__lasx_xvffint_d_lu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve_b(a: m256i, b: i32) -> m256i {
    unsafe { transmute(__lasx_xvreplve_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve_h(a: m256i, b: i32) -> m256i {
    unsafe { transmute(__lasx_xvreplve_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve_w(a: m256i, b: i32) -> m256i {
    unsafe { transmute(__lasx_xvreplve_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve_d(a: m256i, b: i32) -> m256i {
    unsafe { transmute(__lasx_xvreplve_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpermi_w<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvpermi_w(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmuh_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmuh_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_h_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsllwil_h_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_w_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsllwil_w_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_d_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsllwil_d_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_hu_bu<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvsllwil_hu_bu(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_wu_hu<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsllwil_wu_hu(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsllwil_du_wu<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsllwil_du_wu(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsran_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsran_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsran_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsran_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsran_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsran_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_bu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_bu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_hu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_hu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssran_wu_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssran_wu_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarn_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrarn_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarn_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrarn_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarn_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrarn_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_bu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_bu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_hu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_hu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarn_wu_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrarn_wu_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrln_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrln_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrln_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrln_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrln_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrln_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_bu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_bu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_hu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_hu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_wu_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_wu_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrn_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlrn_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrn_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlrn_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrn_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsrlrn_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_bu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_bu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_hu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_hu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_wu_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_wu_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrstpi_b<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvfrstpi_b(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrstpi_h<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvfrstpi_h(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrstp_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvfrstp_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrstp_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvfrstp_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvshuf4i_d<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvshuf4i_d(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbsrl_v<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvbsrl_v(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvbsll_v<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvbsll_v(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextrins_b<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvextrins_b(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextrins_h<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvextrins_h(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextrins_w<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvextrins_w(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextrins_d<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvextrins_d(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmskltz_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmskltz_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmskltz_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmskltz_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmskltz_w(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmskltz_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmskltz_d(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmskltz_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsigncov_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsigncov_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsigncov_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsigncov_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsigncov_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsigncov_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsigncov_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsigncov_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrne_w_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrne_w_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrne_l_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrne_l_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrp_w_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrp_w_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrp_l_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrp_l_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrm_w_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrm_w_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrm_l_d(a: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrm_l_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftint_w_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftint_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffint_s_l(a: m256i, b: m256i) -> m256 {
    unsafe { transmute(__lasx_xvffint_s_l(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrz_w_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrz_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrp_w_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrp_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrm_w_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrm_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrne_w_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvftintrne_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftinth_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftinth_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintl_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintl_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffinth_d_w(a: m256i) -> m256d {
    unsafe { transmute(__lasx_xvffinth_d_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvffintl_d_w(a: m256i) -> m256d {
    unsafe { transmute(__lasx_xvffintl_d_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrzh_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrzh_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrzl_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrzl_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrph_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrph_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrpl_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrpl_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrmh_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrmh_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrml_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrml_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrneh_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrneh_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvftintrnel_l_s(a: m256) -> m256i {
    unsafe { transmute(__lasx_xvftintrnel_l_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrne_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrintrne_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrne_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrintrne_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrz_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrintrz_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrz_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrintrz_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrp_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrintrp_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrp_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrintrp_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrm_s(a: m256) -> m256 {
    unsafe { transmute(__lasx_xvfrintrm_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfrintrm_d(a: m256d) -> m256d {
    unsafe { transmute(__lasx_xvfrintrm_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvld<const IMM_S12: i32>(mem_addr: *const i8) -> m256i {
    static_assert_simm_bits!(IMM_S12, 12);
    transmute(__lasx_xvld(mem_addr, IMM_S12))
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvst<const IMM_S12: i32>(a: m256i, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S12, 12);
    __lasx_xvst(transmute(a), mem_addr, IMM_S12)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_b<const IMM_S8: i32, const IMM4: u32>(a: m256i, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM4, 4);
    __lasx_xvstelm_b(transmute(a), mem_addr, IMM_S8, IMM4)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_h<const IMM_S8: i32, const IMM3: u32>(a: m256i, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM3, 3);
    __lasx_xvstelm_h(transmute(a), mem_addr, IMM_S8, IMM3)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_w<const IMM_S8: i32, const IMM2: u32>(a: m256i, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM2, 2);
    __lasx_xvstelm_w(transmute(a), mem_addr, IMM_S8, IMM2)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2, 3)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstelm_d<const IMM_S8: i32, const IMM1: u32>(a: m256i, mem_addr: *mut i8) {
    static_assert_simm_bits!(IMM_S8, 8);
    static_assert_uimm_bits!(IMM1, 1);
    __lasx_xvstelm_d(transmute(a), mem_addr, IMM_S8, IMM1)
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvinsve0_w<const IMM3: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvinsve0_w(transmute(a), transmute(b), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvinsve0_d<const IMM2: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM2, 2);
    unsafe { transmute(__lasx_xvinsve0_d(transmute(a), transmute(b), IMM2)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickve_w<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvpickve_w(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickve_d<const IMM2: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM2, 2);
    unsafe { transmute(__lasx_xvpickve_d(transmute(a), IMM2)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrn_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrlrn_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_b_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_b_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_h_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_h_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrln_w_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvssrln_w_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvldi<const IMM_S13: i32>() -> m256i {
    static_assert_simm_bits!(IMM_S13, 13);
    unsafe { transmute(__lasx_xvldi(IMM_S13)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldx(mem_addr: *const i8, b: i64) -> m256i {
    transmute(__lasx_xvldx(mem_addr, transmute(b)))
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvstx(a: m256i, mem_addr: *mut i8, b: i64) {
    __lasx_xvstx(transmute(a), mem_addr, transmute(b))
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextl_qu_du(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvextl_qu_du(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve0_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvreplve0_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve0_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvreplve0_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve0_w(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvreplve0_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve0_d(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvreplve0_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvreplve0_q(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvreplve0_q(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_h_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_h_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_w_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_w_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_d_w(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_d_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_w_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_w_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_d_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_d_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_d_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_d_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_hu_bu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_hu_bu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_wu_hu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_wu_hu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_du_wu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_du_wu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_wu_bu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_wu_bu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_du_hu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_du_hu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_vext2xv_du_bu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_vext2xv_du_bu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpermi_q<const IMM8: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvpermi_q(transmute(a), transmute(b), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpermi_d<const IMM8: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(__lasx_xvpermi_d(transmute(a), IMM8)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvperm_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvperm_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_b<const IMM_S12: i32>(mem_addr: *const i8) -> m256i {
    static_assert_simm_bits!(IMM_S12, 12);
    transmute(__lasx_xvldrepl_b(mem_addr, IMM_S12))
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_h<const IMM_S11: i32>(mem_addr: *const i8) -> m256i {
    static_assert_simm_bits!(IMM_S11, 11);
    transmute(__lasx_xvldrepl_h(mem_addr, IMM_S11))
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_w<const IMM_S10: i32>(mem_addr: *const i8) -> m256i {
    static_assert_simm_bits!(IMM_S10, 10);
    transmute(__lasx_xvldrepl_w(mem_addr, IMM_S10))
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn lasx_xvldrepl_d<const IMM_S9: i32>(mem_addr: *const i8) -> m256i {
    static_assert_simm_bits!(IMM_S9, 9);
    transmute(__lasx_xvldrepl_d(mem_addr, IMM_S9))
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwev_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwev_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsubwod_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsubwod_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_d_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_d_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_w_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_w_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_h_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_h_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_q_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_q_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_d_wu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_d_wu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_w_hu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_w_hu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_h_bu(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_h_bu(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_d_wu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_d_wu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_w_hu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_w_hu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_h_bu_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_h_bu_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_d_wu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_d_wu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_w_hu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_w_hu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_h_bu_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_h_bu_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_d_wu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_d_wu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_w_hu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_w_hu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_h_bu_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_h_bu_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_d_wu_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_d_wu_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_w_hu_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_w_hu_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_h_bu_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_h_bu_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhaddw_qu_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhaddw_qu_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_q_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_q_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvhsubw_qu_du(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvhsubw_qu_du(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_q_d(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_q_d(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_d_w(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_d_w(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_w_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_w_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_h_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_h_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_q_du(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_q_du(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_d_wu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_d_wu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_w_hu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_w_hu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_h_bu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_h_bu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_q_d(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_q_d(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_d_w(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_d_w(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_w_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_w_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_h_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_h_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_q_du(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_q_du(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_d_wu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_d_wu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_w_hu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_w_hu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_h_bu(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_h_bu(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_q_du_d(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_q_du_d(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_d_wu_w(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_d_wu_w(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_w_hu_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_w_hu_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwev_h_bu_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwev_h_bu_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_q_du_d(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_q_du_d(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_d_wu_w(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_d_wu_w(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_w_hu_h(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_w_hu_h(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmaddwod_h_bu_b(a: m256i, b: m256i, c: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmaddwod_h_bu_b(transmute(a), transmute(b), transmute(c))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotr_b(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvrotr_b(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotr_h(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvrotr_h(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotr_w(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvrotr_w(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotr_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvrotr_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvadd_q(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvadd_q(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsub_q(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvsub_q(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwev_q_du_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwev_q_du_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvaddwod_q_du_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvaddwod_q_du_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwev_q_du_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwev_q_du_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmulwod_q_du_d(a: m256i, b: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmulwod_q_du_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmskgez_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmskgez_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvmsknz_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvmsknz_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_h_b(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_h_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_w_h(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_w_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_d_w(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_d_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_q_d(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_q_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_hu_bu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_hu_bu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_wu_hu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_wu_hu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_du_wu(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_du_wu(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvexth_qu_du(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvexth_qu_du(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotri_b<const IMM3: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvrotri_b(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotri_h<const IMM4: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvrotri_h(transmute(a), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotri_w<const IMM5: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvrotri_w(transmute(a), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvrotri_d<const IMM6: u32>(a: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvrotri_d(transmute(a), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvextl_q_d(a: m256i) -> m256i {
    unsafe { transmute(__lasx_xvextl_q_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrlni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrlni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrlni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvsrlni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrlrni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrlrni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrlrni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrlrni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvsrlrni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrlni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrlni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrlni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrlni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_bu_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrlni_bu_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_hu_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrlni_hu_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_wu_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrlni_wu_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlni_du_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrlni_du_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrlrni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrlrni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrlrni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrlrni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_bu_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrlrni_bu_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_hu_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrlrni_hu_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_wu_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrlrni_wu_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrlrni_du_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrlrni_du_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrani_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrani_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrani_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrani_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrani_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrani_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrani_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvsrani_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvsrarni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvsrarni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvsrarni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvsrarni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvsrarni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrani_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrani_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrani_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrani_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_bu_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrani_bu_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_hu_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrani_hu_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_wu_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrani_wu_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrani_du_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrani_du_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_b_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrarni_b_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_h_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrarni_h_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_w_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrarni_w_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_d_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrarni_d_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_bu_h<const IMM4: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM4, 4);
    unsafe { transmute(__lasx_xvssrarni_bu_h(transmute(a), transmute(b), IMM4)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_hu_w<const IMM5: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM5, 5);
    unsafe { transmute(__lasx_xvssrarni_hu_w(transmute(a), transmute(b), IMM5)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_wu_d<const IMM6: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM6, 6);
    unsafe { transmute(__lasx_xvssrarni_wu_d(transmute(a), transmute(b), IMM6)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvssrarni_du_q<const IMM7: u32>(a: m256i, b: m256i) -> m256i {
    static_assert_uimm_bits!(IMM7, 7);
    unsafe { transmute(__lasx_xvssrarni_du_q(transmute(a), transmute(b), IMM7)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbnz_b(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbnz_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbnz_d(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbnz_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbnz_h(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbnz_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbnz_v(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbnz_v(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbnz_w(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbnz_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbz_b(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbz_b(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbz_d(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbz_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbz_h(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbz_h(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbz_v(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbz_v(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xbz_w(a: m256i) -> i32 {
    unsafe { transmute(__lasx_xbz_w(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_caf_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_caf_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_caf_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_caf_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_ceq_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_ceq_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_ceq_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_ceq_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cle_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cle_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cle_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cle_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_clt_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_clt_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_clt_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_clt_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cne_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cne_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cne_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cne_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cor_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cor_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cor_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cor_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cueq_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cueq_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cueq_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cueq_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cule_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cule_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cule_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cule_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cult_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cult_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cult_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cult_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cun_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cun_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cune_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cune_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cune_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cune_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_cun_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_cun_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_saf_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_saf_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_saf_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_saf_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_seq_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_seq_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_seq_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_seq_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sle_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sle_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sle_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sle_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_slt_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_slt_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_slt_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_slt_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sne_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sne_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sne_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sne_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sor_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sor_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sor_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sor_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sueq_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sueq_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sueq_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sueq_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sule_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sule_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sule_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sule_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sult_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sult_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sult_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sult_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sun_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sun_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sune_d(a: m256d, b: m256d) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sune_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sune_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sune_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvfcmp_sun_s(a: m256, b: m256) -> m256i {
    unsafe { transmute(__lasx_xvfcmp_sun_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickve_d_f<const IMM2: u32>(a: m256d) -> m256d {
    static_assert_uimm_bits!(IMM2, 2);
    unsafe { transmute(__lasx_xvpickve_d_f(transmute(a), IMM2)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_xvpickve_w_f<const IMM3: u32>(a: m256) -> m256 {
    static_assert_uimm_bits!(IMM3, 3);
    unsafe { transmute(__lasx_xvpickve_w_f(transmute(a), IMM3)) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_cast_128_s(a: m128) -> m256 {
    unsafe { transmute(__lasx_cast_128_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_cast_128_d(a: m128d) -> m256d {
    unsafe { transmute(__lasx_cast_128_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_cast_128(a: m128i) -> m256i {
    unsafe { transmute(__lasx_cast_128(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_concat_128_s(a: m128, b: m128) -> m256 {
    unsafe { transmute(__lasx_concat_128_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_concat_128_d(a: m128d, b: m128d) -> m256d {
    unsafe { transmute(__lasx_concat_128_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_concat_128(a: m128i, b: m128i) -> m256i {
    unsafe { transmute(__lasx_concat_128(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_lo_s(a: m256) -> m128 {
    unsafe { transmute(__lasx_extract_128_lo_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_hi_s(a: m256) -> m128 {
    unsafe { transmute(__lasx_extract_128_hi_s(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_lo_d(a: m256d) -> m128d {
    unsafe { transmute(__lasx_extract_128_lo_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_hi_d(a: m256d) -> m128d {
    unsafe { transmute(__lasx_extract_128_hi_d(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_lo(a: m256i) -> m128i {
    unsafe { transmute(__lasx_extract_128_lo(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_extract_128_hi(a: m256i) -> m128i {
    unsafe { transmute(__lasx_extract_128_hi(transmute(a))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_lo_s(a: m256, b: m128) -> m256 {
    unsafe { transmute(__lasx_insert_128_lo_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_hi_s(a: m256, b: m128) -> m256 {
    unsafe { transmute(__lasx_insert_128_hi_s(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_lo_d(a: m256d, b: m128d) -> m256d {
    unsafe { transmute(__lasx_insert_128_lo_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_hi_d(a: m256d, b: m128d) -> m256d {
    unsafe { transmute(__lasx_insert_128_hi_d(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_lo(a: m256i, b: m128i) -> m256i {
    unsafe { transmute(__lasx_insert_128_lo(transmute(a), transmute(b))) }
}

#[inline]
#[target_feature(enable = "lasx")]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub fn lasx_insert_128_hi(a: m256i, b: m128i) -> m256i {
    unsafe { transmute(__lasx_insert_128_hi(transmute(a), transmute(b))) }
}
