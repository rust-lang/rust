use abi::errors::{errno, Errno};
use abi::syscall::{SYS_VM_ADVISE, SYS_VM_MAP, SYS_VM_PROTECT, SYS_VM_QUERY, SYS_VM_UNMAP};
pub use abi::vm::*;

use crate::syscall::syscall6;

pub fn vm_map(req: &VmMapReq) -> Result<VmMapResp, Errno> {
    let mut resp = VmMapResp::default();
    let ret = unsafe {
        syscall6(
            SYS_VM_MAP,
            req as *const VmMapReq as usize,
            &mut resp as *mut VmMapResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    errno(ret).map(|_| resp)
}

pub fn vm_unmap(req: &VmUnmapReq) -> Result<VmUnmapResp, Errno> {
    let mut resp = VmUnmapResp::default();
    let ret = unsafe {
        syscall6(
            SYS_VM_UNMAP,
            req as *const VmUnmapReq as usize,
            &mut resp as *mut VmUnmapResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    errno(ret).map(|_| resp)
}

pub fn vm_protect(req: &VmProtectReq) -> Result<(), Errno> {
    let ret = unsafe {
        syscall6(
            SYS_VM_PROTECT,
            req as *const VmProtectReq as usize,
            0,
            0,
            0,
            0,
            0,
        )
    };
    errno(ret).map(|_| ())
}

pub fn vm_advise(req: &VmAdviseReq) -> Result<(), Errno> {
    let ret = unsafe {
        syscall6(
            SYS_VM_ADVISE,
            req as *const VmAdviseReq as usize,
            0,
            0,
            0,
            0,
            0,
        )
    };
    errno(ret).map(|_| ())
}

pub fn vm_query(req: &VmQueryReq) -> Result<VmQueryResp, Errno> {
    let mut resp = VmQueryResp::default();
    let ret = unsafe {
        syscall6(
            SYS_VM_QUERY,
            req as *const VmQueryReq as usize,
            &mut resp as *mut VmQueryResp as usize,
            0,
            0,
            0,
            0,
        )
    };
    errno(ret).map(|_| resp)
}
