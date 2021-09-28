//! ABI for μITRON derivatives
pub type int_t = crate::os::raw::c_int;
pub type uint_t = crate::os::raw::c_uint;
pub type bool_t = int_t;

/// Kernel object ID
pub type ID = int_t;

/// The current task.
pub const TSK_SELF: ID = 0;

/// Relative time
pub type RELTIM = u32;

/// Timeout (a valid `RELTIM` value or `TMO_FEVR`)
pub type TMO = u32;

/// The infinite timeout value
pub const TMO_FEVR: TMO = TMO::MAX;

/// The maximum valid value of `RELTIM`
pub const TMAX_RELTIM: RELTIM = 4_000_000_000;

/// System time
pub type SYSTIM = u64;

/// Error code type
pub type ER = int_t;

/// Error code type, `ID` on success
pub type ER_ID = int_t;

/// Task or interrupt priority
pub type PRI = int_t;

/// The special value of `PRI` representing the current task's priority.
pub const TPRI_SELF: PRI = 0;

/// Object attributes
pub type ATR = uint_t;

/// Use the priority inheritance protocol
#[cfg(target_os = "solid_asp3")]
pub const TA_INHERIT: ATR = 0x02;

/// Activate the task on creation
pub const TA_ACT: ATR = 0x01;

/// The maximum count of a semaphore
pub const TMAX_MAXSEM: uint_t = uint_t::MAX;

/// Callback parameter
pub type EXINF = isize;

/// Task entrypoint
pub type TASK = Option<unsafe extern "C" fn(EXINF)>;

// Error codes
pub const E_OK: ER = 0;
pub const E_SYS: ER = -5;
pub const E_NOSPT: ER = -9;
pub const E_RSFN: ER = -10;
pub const E_RSATR: ER = -11;
pub const E_PAR: ER = -17;
pub const E_ID: ER = -18;
pub const E_CTX: ER = -25;
pub const E_MACV: ER = -26;
pub const E_OACV: ER = -27;
pub const E_ILUSE: ER = -28;
pub const E_NOMEM: ER = -33;
pub const E_NOID: ER = -34;
pub const E_NORES: ER = -35;
pub const E_OBJ: ER = -41;
pub const E_NOEXS: ER = -42;
pub const E_QOVR: ER = -43;
pub const E_RLWAI: ER = -49;
pub const E_TMOUT: ER = -50;
pub const E_DLT: ER = -51;
pub const E_CLS: ER = -52;
pub const E_RASTER: ER = -53;
pub const E_WBLK: ER = -57;
pub const E_BOVR: ER = -58;
pub const E_COMM: ER = -65;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct T_CSEM {
    pub sematr: ATR,
    pub isemcnt: uint_t,
    pub maxsem: uint_t,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct T_CMTX {
    pub mtxatr: ATR,
    pub ceilpri: PRI,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct T_CTSK {
    pub tskatr: ATR,
    pub exinf: EXINF,
    pub task: TASK,
    pub itskpri: PRI,
    pub stksz: usize,
    pub stk: *mut u8,
}

extern "C" {
    #[link_name = "__asp3_acre_tsk"]
    pub fn acre_tsk(pk_ctsk: *const T_CTSK) -> ER_ID;
    #[link_name = "__asp3_get_tid"]
    pub fn get_tid(p_tskid: *mut ID) -> ER;
    #[link_name = "__asp3_dly_tsk"]
    pub fn dly_tsk(dlytim: RELTIM) -> ER;
    #[link_name = "__asp3_ter_tsk"]
    pub fn ter_tsk(tskid: ID) -> ER;
    #[link_name = "__asp3_del_tsk"]
    pub fn del_tsk(tskid: ID) -> ER;
    #[link_name = "__asp3_get_pri"]
    pub fn get_pri(tskid: ID, p_tskpri: *mut PRI) -> ER;
    #[link_name = "__asp3_rot_rdq"]
    pub fn rot_rdq(tskpri: PRI) -> ER;
    #[link_name = "__asp3_slp_tsk"]
    pub fn slp_tsk() -> ER;
    #[link_name = "__asp3_tslp_tsk"]
    pub fn tslp_tsk(tmout: TMO) -> ER;
    #[link_name = "__asp3_wup_tsk"]
    pub fn wup_tsk(tskid: ID) -> ER;
    #[link_name = "__asp3_unl_cpu"]
    pub fn unl_cpu() -> ER;
    #[link_name = "__asp3_dis_dsp"]
    pub fn dis_dsp() -> ER;
    #[link_name = "__asp3_ena_dsp"]
    pub fn ena_dsp() -> ER;
    #[link_name = "__asp3_sns_dsp"]
    pub fn sns_dsp() -> bool_t;
    #[link_name = "__asp3_get_tim"]
    pub fn get_tim(p_systim: *mut SYSTIM) -> ER;
    #[link_name = "__asp3_acre_mtx"]
    pub fn acre_mtx(pk_cmtx: *const T_CMTX) -> ER_ID;
    #[link_name = "__asp3_del_mtx"]
    pub fn del_mtx(tskid: ID) -> ER;
    #[link_name = "__asp3_loc_mtx"]
    pub fn loc_mtx(mtxid: ID) -> ER;
    #[link_name = "__asp3_ploc_mtx"]
    pub fn ploc_mtx(mtxid: ID) -> ER;
    #[link_name = "__asp3_tloc_mtx"]
    pub fn tloc_mtx(mtxid: ID, tmout: TMO) -> ER;
    #[link_name = "__asp3_unl_mtx"]
    pub fn unl_mtx(mtxid: ID) -> ER;
    pub fn exd_tsk() -> ER;
}
