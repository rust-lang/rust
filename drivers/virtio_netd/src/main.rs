//! VirtIO-NET userspace driver
//!
//! This service owns the VirtIO-NET device hardware and exposes it as a VFS
//! provider mounted at `/dev/net/virtio0/`.
//!
//! Architecture:
//! - virtio_netd: Hardware driver (RX/TX queues, DMA buffers, interrupts)
//!   Exposes files: ctl, status, mac, mtu, rx, tx, features, events
//! - netd / other consumers: talk to the driver purely through file paths
#![no_std]
#![no_main]
extern crate alloc;
use alloc::string::{String, ToString};



mod driver;
mod vfs_provider;

use abi::vfs_rpc::VFS_RPC_MAX_REQ;
use alloc::vec;
use driver::VirtioNetDriver;
use stem::syscall::{channel_create, channel_try_recv};
use stem::{error, warn};
use vfs_provider::{handle_vfs_rpc, NetVfsState};

#[stem::main]
fn main(arg: usize) -> ! {
    stem::debug!("VIRTIO_NETD: Starting VirtIO-NET driver service...");

    // 1. Map bootstrap memfd
    let mut drv_req_read = 0;
    let mut drv_resp_write = 0;
    let mut _supervisor_port = 0;
    let mut bind_instance_id = 0u64;
    let mut claimed_path = String::new();

    if arg != 0 {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: 4096,
            prot: VmProt::READ | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: arg as u32,
                offset: 0,
            },
        };
        if let Ok(resp) = stem::syscall::vm_map(&req) {
            let slice = unsafe { core::slice::from_raw_parts(resp.addr as *const u32, 1024) };

            drv_req_read = slice[0];
            drv_resp_write = slice[1];
            _supervisor_port = slice[2];

            let id_low = slice[3] as u64;
            let id_high = slice[4] as u64;
            bind_instance_id = id_low | (id_high << 32);

            // Path is at offset 512 bytes (index 128 in u32 slice)
            let path_bytes =
                unsafe { core::slice::from_raw_parts((resp.addr + 512) as *const u8, 128) };
            let path_len = path_bytes.iter().position(|&b| b == 0).unwrap_or(128);
            claimed_path = core::str::from_utf8(&path_bytes[..path_len])
                .unwrap_or("")
                .to_string();

            stem::debug!("VIRTIO_NETD: Bootstrap handles: req_read={}, resp_write={}, id={}, path={}", 
                drv_req_read, drv_resp_write, bind_instance_id, claimed_path);
        } else {
            warn!("VIRTIO_NETD: Failed to map bootstrap memfd!");
        }
    }

    stem::debug!("VIRTIO_NETD: Initializing hardware driver...");

    // Initialize VirtIO-NET driver.
    let mut driver: VirtioNetDriver = match if !claimed_path.is_empty() {
        VirtioNetDriver::claim_device(&claimed_path)
    } else {
        VirtioNetDriver::find_and_claim()
    } {
        Ok(d) => {
            stem::debug!("VIRTIO_NETD: Driver initialized successfully");
            d
        }
        Err(e) => {
            error!("VIRTIO_NETD: Failed to initialize driver: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    let mac = driver.mac();
    stem::debug!(
        "VIRTIO_NETD: MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0],
        mac[1],
        mac[2],
        mac[3],
        mac[4],
        mac[5]
    );

    // Wait for link up before proceeding.
    stem::debug!("VIRTIO_NETD: Waiting for link...");
    loop {
        if driver.link_up() {
            stem::debug!("VIRTIO_NETD: Link is UP");
            break;
        }
        stem::time::sleep_ms(100);
    }

    let features = driver.device_features();

    // Create the VFS provider port pair.
    //   req_write → kernel sends VFS RPCs here
    //   req_read  → this daemon reads RPCs here
    let (req_write, req_read) = match channel_create(VFS_RPC_MAX_REQ * 8) {
        Ok(handles) => {
            stem::debug!("VIRTIO_NETD: Created VFS provider port");
            handles
        }
        Err(e) => {
            error!("VIRTIO_NETD: Failed to create provider port: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    // Sovereign Handshake
    use abi::display_driver_protocol;
    use abi::supervisor_protocol::{self, classes}; // Still used for common header

    let ready = supervisor_protocol::BindReadyPayload {
        bind_instance_id,
        class_mask: classes::NETWORK_INTERFACE,
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
            // Bundle the VFS provider handle and the BIND_READY notification atomically.
            let _ = stem::syscall::channel::channel_send_msg(
                drv_resp_write,
                &buf[..total_len],
                &[req_write],
            );
            stem::debug!("VIRTIO_NETD: Sent MSG_BIND_READY, waiting for MSG_BIND_ASSIGNED...");
        }
    }

    // Wait for MSG_BIND_ASSIGNED or MSG_BIND_FAILED
    let mut wait_buf = [0u8; 512];
    let assigned_bind_id = loop {
        if let Ok(n) = stem::syscall::channel_try_recv(drv_req_read, &mut wait_buf) {
            if let Some((header, payload)) = display_driver_protocol::parse_message(&wait_buf[..n])
            {
                if header.msg_type == supervisor_protocol::MSG_BIND_ASSIGNED {
                    if let Some(assigned) = supervisor_protocol::decode_bind_assigned_le(payload) {
                        let path_len = assigned
                            .primary_path
                            .iter()
                            .position(|&b| b == 0)
                            .unwrap_or(64);
                        let path =
                            core::str::from_utf8(&assigned.primary_path[..path_len]).unwrap_or("?");
                        stem::debug!(
                            "VIRTIO_NETD: Sovereign registration COMPLETE. Assigned: {}",
                            path
                        );
                        break assigned.bind_instance_id;
                    }
                } else if header.msg_type == supervisor_protocol::MSG_BIND_FAILED {
                    if let Some(failed) = supervisor_protocol::decode_bind_failed_le(payload) {
                        let reason_len = failed.reason.iter().position(|&b| b == 0).unwrap_or(64);
                        let reason =
                            core::str::from_utf8(&failed.reason[..reason_len]).unwrap_or("?");
                        stem::warn!("VIRTIO_NETD: Registration REJECTED by supervisor (code={}, reason={}). Halting.", failed.error_code, reason);
                        loop {
                            stem::syscall::yield_now();
                        }
                    }
                }
            }
        }
        stem::syscall::yield_now();
    };

    // Notify supervisor that this service is fully operational.
    {
        let svc_ready = supervisor_protocol::ServiceReadyPayload {
            bind_instance_id: assigned_bind_id,
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
                stem::debug!("VIRTIO_NETD: Sent MSG_SERVICE_READY.");
            }
        }
    }

    // Initialize shared VFS state.
    let mut state = NetVfsState::new(mac, true, features);

    // Main loop: interleave hardware polling with VFS RPC handling.
    let mut req_buf = vec![0u8; VFS_RPC_MAX_REQ];
    loop {
        // 1. Poll for link-state changes and queue events.
        if let Some(link_up) = driver.poll_link_change() {
            state.link_up = link_up;
            let event = if link_up { "link-up" } else { "link-down" };
            state.push_event(event);
            stem::debug!("VIRTIO_NETD: Link state changed: {}", event);
        }

        // 2. Poll hardware for received frames and buffer them.
        if let Some(frame) = driver.poll_rx() {
            let frame_vec: alloc::vec::Vec<u8> = frame.to_vec();
            state.push_rx_frame(frame_vec);
        }

        // 3. Service any pending VFS RPC (non-blocking).
        match channel_try_recv(req_read, &mut req_buf) {
            Ok(n) if n > 0 => {
                handle_vfs_rpc(&mut state, &mut driver, &req_buf[..n]);
            }
            Err(e) if e != abi::errors::Errno::EAGAIN => {
                warn!("VIRTIO_NETD: port_try_recv error: {:?}", e);
            }
            _ => {}
        }

        stem::time::sleep_ms(1);
    }
}
