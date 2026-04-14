#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


mod driver;
mod vfs_provider;

use abi::vfs_rpc::VFS_RPC_MAX_REQ;
use driver::BootFbDriver;
use ipc_helpers::provider::ProviderLoop;
use stem::syscall::{channel_create, channel_recv};
use stem::{debug, info, warn};
use vfs_provider::dispatch_vfs_rpc;

#[stem::main]
fn main(boot_fd: usize) -> ! {
    debug!("display_bootfb: Starting VFS-native bootfb driver...");
    debug!("display_bootfb: Liveness check: driver is alive.");

    // 1. Map bootstrap memfd to get handles
    let mut drv_req_read = 0;
    let mut drv_resp_write = 0;
    let mut supervisor_port = 0;
    let mut bind_instance_id = 0u64;

    let mut boot_fd = boot_fd;

    if boot_fd == 0 {
        let mut buf = [0u8; 1024];
        if let Ok(needed) = stem::syscall::argv_get(&mut buf) {
            if needed >= 4 {
                let count = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                if count >= 2 {
                    let mut offset = 4;
                    // Skip argv[0]
                    let arg0_len =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4 + arg0_len;
                    // argv[1]
                    if offset + 4 <= buf.len() {
                        let arg1_len =
                            u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
                                as usize;
                        offset += 4;
                        if offset + arg1_len <= buf.len() {
                            if let Ok(s) = core::str::from_utf8(&buf[offset..offset + arg1_len]) {
                                if let Ok(val) = s.parse::<usize>() {
                                    boot_fd = val;
                                    debug!(
                                        "display_bootfb: Recovered boot_fd {} from argv[1]",
                                        boot_fd
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if boot_fd != 0 {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: 4096,
            prot: VmProt::READ | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: boot_fd as u32,
                offset: 0,
            },
        };
        if let Ok(resp) = stem::syscall::vm_map(&req) {
            let slice = unsafe { core::slice::from_raw_parts(resp.addr as *const u32, 1024) };

            // Layout from sprout/src/pipelines.rs:
            // slice[0]: drv_req_read
            // slice[1]: drv_resp_write
            // slice[2]: supervisor_port
            // slice[3..5]: bind_instance_id (u64)

            drv_req_read = slice[0];
            drv_resp_write = slice[1];
            supervisor_port = slice[2];

            let id_low = slice[3] as u64;
            let id_high = slice[4] as u64;
            bind_instance_id = id_low | (id_high << 32);

            debug!(
                "display_bootfb: Bootstrap handles: req_read={}, resp_write={}, svc={}, id={}",
                drv_req_read, drv_resp_write, supervisor_port, bind_instance_id
            );
        } else {
            warn!("display_bootfb: Failed to map bootstrap memfd!");
        }
    } else {
        stem::debug!("display_bootfb: ERROR: No bootstrap memfd arg provided");
    }

    if drv_req_read == 0 || drv_resp_write == 0 || supervisor_port == 0 || bind_instance_id == 0 {
        stem::debug!("display_bootfb: ERROR: Invalid/Missing bootstrap components (req={}, resp={}, svc={}, id={})", 
            drv_req_read, drv_resp_write, supervisor_port, bind_instance_id);
        loop {
            stem::yield_now();
        }
    }

    let mut driver = match BootFbDriver::new() {
        Some(d) => {
            debug!(
                "display_bootfb: Driver initialized successfully ({}x{})",
                d.fb.width, d.fb.height
            );
            debug!(
                "display_bootfb: Mapping framebuffer (backing fd={})...",
                boot_fd
            );
            d
        }
        None => {
            debug!("display_bootfb: ERROR: Failed to acquire hardware framebuffer");
            loop {
                stem::yield_now();
            }
        }
    };

    // Create the VFS provider port pair.
    let (vfs_write, vfs_read) = match channel_create(VFS_RPC_MAX_REQ * 8) {
        Ok(handles) => handles,
        Err(e) => {
            debug!(
                "display_bootfb: ERROR: Failed to create provider port: {:?}",
                e
            );
            loop {
                stem::yield_now();
            }
        }
    };

    // Sovereign Handshake
    use abi::display_driver_protocol;
    use abi::supervisor_protocol::{self, classes};

    let ready = supervisor_protocol::BindReadyPayload {
        bind_instance_id,
        class_mask: classes::DISPLAY_CARD | classes::FRAMEBUFFER,
        _reserved: 0,
    };
    let mut ready_bytes = [0u8; supervisor_protocol::BIND_READY_PAYLOAD_SIZE];
    if let Some(len) = supervisor_protocol::encode_bind_ready_le(&ready, &mut ready_bytes) {
        let mut buf = [0u8; 256];
        if let Some(total_len) = display_driver_protocol::encode_message(
            &mut buf,
            supervisor_protocol::MSG_BIND_READY,
            &ready_bytes[..len],
        ) {
            debug!("display_bootfb: Sending MSG_BIND_READY handshake...");
            // Bundle the VFS provider handle and the BIND_READY notification atomically.
            let _ = stem::syscall::channel::channel_send_msg(
                drv_resp_write,
                &buf[..total_len],
                &[vfs_write],
            );
            debug!("display_bootfb: Sent MSG_BIND_READY, waiting for MSG_BIND_ASSIGNED...");
        }
    }

    // Wait for MSG_BIND_ASSIGNED or MSG_BIND_FAILED
    let mut wait_buf = [0u8; 512];
    let mut bind_instance_id_confirmed = bind_instance_id;
    loop {
        // Read from drv_req_read, NOT supervisor_port. This should block rather than
        // spin so the CPU can schedule unrelated work while the driver waits.
        match channel_recv(drv_req_read, &mut wait_buf) {
            Ok(n) => {
                if let Some((header, payload)) = display_driver_protocol::parse_message(&wait_buf[..n])
                {
                    if header.msg_type == supervisor_protocol::MSG_BIND_ASSIGNED {
                        if let Some(assigned) = supervisor_protocol::decode_bind_assigned_le(payload)
                        {
                            bind_instance_id_confirmed = assigned.bind_instance_id;
                            let path_len = assigned
                                .primary_path
                                .iter()
                                .position(|&b| b == 0)
                                .unwrap_or(64);
                            let path =
                                core::str::from_utf8(&assigned.primary_path[..path_len]).unwrap_or("?");
                            debug!(
                                "display_bootfb: Sovereign registration COMPLETE. Assigned: {}",
                                path
                            );
                            break;
                        }
                    } else if header.msg_type == supervisor_protocol::MSG_BIND_FAILED {
                        if let Some(failed) = supervisor_protocol::decode_bind_failed_le(payload) {
                            let reason_len = failed.reason.iter().position(|&b| b == 0).unwrap_or(64);
                            let reason =
                                core::str::from_utf8(&failed.reason[..reason_len]).unwrap_or("?");
                            warn!("display_bootfb: Registration REJECTED by supervisor (code={}, reason={}). Halting.", failed.error_code, reason);
                            loop {
                                stem::syscall::yield_now();
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!("display_bootfb: failed waiting for bind assignment: {:?}", e);
                stem::time::sleep_ms(1);
            }
        }
    }

    // Notify supervisor that this service is now fully operational.
    {
        let svc_ready = supervisor_protocol::ServiceReadyPayload {
            bind_instance_id: bind_instance_id_confirmed,
            _reserved: 0,
        };
        let mut payload_bytes = [0u8; supervisor_protocol::SERVICE_READY_PAYLOAD_SIZE];
        let mut svc_buf = [0u8; 64];
        if let Some(p_len) =
            supervisor_protocol::encode_service_ready_le(&svc_ready, &mut payload_bytes)
        {
            if let Some(total_len) = display_driver_protocol::encode_message(
                &mut svc_buf,
                supervisor_protocol::MSG_SERVICE_READY,
                &payload_bytes[..p_len],
            ) {
                let _ = stem::syscall::channel::channel_send_msg(
                    drv_resp_write,
                    &svc_buf[..total_len],
                    &[],
                );
                debug!("display_bootfb: Sent MSG_SERVICE_READY.");
            }
        }
    }

    // VFS provider service loop — ProviderLoop blocks on channel_recv and
    // dispatches each decoded request to dispatch_vfs_rpc.
    info!("display_bootfb: entering VFS provider service loop");
    let mut lp = ProviderLoop::new(vfs_read);
    loop {
        let req = match lp.next_request() {
            Ok(r) => r,
            Err(_) => break,
        };
        let resp = dispatch_vfs_rpc(&mut driver, &req);
        lp.send_response(req.resp_port, resp).ok();
    }

    info!("display_bootfb: VFS provider channel closed — halting");
    loop {
        stem::syscall::yield_now();
    }
}
