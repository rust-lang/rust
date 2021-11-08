use crate::os::raw::c_int;

mod fs;
pub mod sockets;
pub use self::fs::*;

#[inline(always)]
pub fn breakpoint_program_exited(tid: usize) {
    unsafe {
        match () {
            // SOLID_BP_PROGRAM_EXITED = 15
            #[cfg(target_arch = "arm")]
            () => asm!("bkpt #15", in("r0") tid),
            #[cfg(target_arch = "aarch64")]
            () => asm!("hlt #15", in("x0") tid),
        }
    }
}

#[inline(always)]
pub fn breakpoint_abort() {
    unsafe {
        match () {
            // SOLID_BP_CSABORT = 16
            #[cfg(target_arch = "arm")]
            () => asm!("bkpt #16"),
            #[cfg(target_arch = "aarch64")]
            () => asm!("hlt #16"),
        }
    }
}

// `solid_types.h`
pub use super::itron::abi::{ER, ER_ID, E_TMOUT, ID};

pub const SOLID_ERR_NOTFOUND: ER = -1000;
pub const SOLID_ERR_NOTSUPPORTED: ER = -1001;
pub const SOLID_ERR_EBADF: ER = -1002;
pub const SOLID_ERR_INVALIDCONTENT: ER = -1003;
pub const SOLID_ERR_NOTUSED: ER = -1004;
pub const SOLID_ERR_ALREADYUSED: ER = -1005;
pub const SOLID_ERR_OUTOFBOUND: ER = -1006;
pub const SOLID_ERR_BADSEQUENCE: ER = -1007;
pub const SOLID_ERR_UNKNOWNDEVICE: ER = -1008;
pub const SOLID_ERR_BUSY: ER = -1009;
pub const SOLID_ERR_TIMEOUT: ER = -1010;
pub const SOLID_ERR_INVALIDACCESS: ER = -1011;
pub const SOLID_ERR_NOTREADY: ER = -1012;

// `solid_rtc.h`
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SOLID_RTC_TIME {
    pub tm_sec: c_int,
    pub tm_min: c_int,
    pub tm_hour: c_int,
    pub tm_mday: c_int,
    pub tm_mon: c_int,
    pub tm_year: c_int,
    pub tm_wday: c_int,
}

extern "C" {
    pub fn SOLID_RTC_ReadTime(time: *mut SOLID_RTC_TIME) -> c_int;
}

// `solid_log.h`
extern "C" {
    pub fn SOLID_LOG_write(s: *const u8, l: usize);
}

// `solid_mem.h`
extern "C" {
    pub fn SOLID_TLS_AddDestructor(id: i32, dtor: unsafe extern "C" fn(*mut u8));
}

// `solid_rng.h`
extern "C" {
    pub fn SOLID_RNG_SampleRandomBytes(buffer: *mut u8, length: usize) -> c_int;
}

// `rwlock.h`
extern "C" {
    pub fn rwl_loc_rdl(id: ID) -> ER;
    pub fn rwl_loc_wrl(id: ID) -> ER;
    pub fn rwl_ploc_rdl(id: ID) -> ER;
    pub fn rwl_ploc_wrl(id: ID) -> ER;
    pub fn rwl_unl_rwl(id: ID) -> ER;
    pub fn rwl_acre_rwl() -> ER_ID;
    pub fn rwl_del_rwl(id: ID) -> ER;
}
