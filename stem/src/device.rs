use crate::syscall::syscall6;
use abi::device::{
    DeviceCall, DeviceKind, PciEnableMsiRequest, PciEnableMsiResponse, RtcTime, PCI_OP_ENABLE_MSI,
    RTC_OP_READ_TIME,
};
use abi::errors::Errno;
use abi::syscall::SYS_DEVICE_CALL;

pub fn device_call(call: &mut DeviceCall) -> Result<usize, Errno> {
    let ret = unsafe { syscall6(SYS_DEVICE_CALL, call as *mut _ as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret)
}

pub fn rtc_read_time() -> Result<RtcTime, Errno> {
    let mut time = RtcTime::default();
    let mut call = DeviceCall {
        kind: DeviceKind::RtcCmos,
        op: RTC_OP_READ_TIME,
        in_ptr: 0,
        in_len: 0,
        out_ptr: &mut time as *mut _ as u64,
        out_len: core::mem::size_of::<RtcTime>() as u32,
    };

    device_call(&mut call)?;
    Ok(time)
}

pub fn device_enable_msi(
    claim_handle: usize,
    prefer_msix: bool,
) -> Result<PciEnableMsiResponse, Errno> {
    let mut response = PciEnableMsiResponse {
        vector: 0,
        irq_mode: 0,
        _reserved: [0; 2],
    };
    let req = PciEnableMsiRequest {
        claim_handle: claim_handle as u32,
        requested_vectors: 1,
        prefer_msix: prefer_msix as u8,
        _reserved: 0,
    };
    let mut call = DeviceCall {
        kind: DeviceKind::Pci,
        op: PCI_OP_ENABLE_MSI,
        in_ptr: &req as *const _ as u64,
        in_len: core::mem::size_of::<PciEnableMsiRequest>() as u32,
        out_ptr: &mut response as *mut _ as u64,
        out_len: core::mem::size_of::<PciEnableMsiResponse>() as u32,
    };

    device_call(&mut call)?;
    Ok(response)
}
