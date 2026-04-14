#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


mod driver;
mod protocol;

use driver::Rtl8168Driver;
use protocol::{NetDriverMsg, MSG_FRAME_RX, MSG_FRAME_TX, MSG_MAC_REQ, MSG_MAC_RESP};
use stem::syscall::{channel_create, channel_recv, channel_send};
use stem::{error, info, warn};

const KIND_NET_DRIVER: &str = "svc.net.Driver";

#[stem::main]
fn main(boot_fd: usize) -> ! {
    info!(
        "RTL8168D: starting Realtek RTL8111/8168 driver (boot_fd={})",
        boot_fd
    );

    // 1. Get device path from bootstrap memfd
    let path = if boot_fd != 0 {
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
            let ptr = resp.addr as *const u8;
            let len = (0..128)
                .find(|&i| unsafe { *ptr.add(i) == 0 })
                .unwrap_or(128);
            unsafe { core::slice::from_raw_parts(ptr, len) }
        } else {
            b"/sys/devices/pci-02:01.0" // Placeholder
        }
    } else {
        b"/sys/devices/pci-02:01.0"
    };
    let path_str = core::str::from_utf8(path).unwrap_or("");

    let mut driver = match Rtl8168Driver::new(path_str) {
        Ok(d) => d,
        Err(e) => {
            error!("RTL8168D: init failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    let mac = driver.mac();
    info!(
        "RTL8168D: MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, link={}",
        mac[0],
        mac[1],
        mac[2],
        mac[3],
        mac[4],
        mac[5],
        if driver.link_up() { "up" } else { "down" }
    );

    let (tx_write, tx_read) = match channel_create(65536) {
        Ok(h) => h,
        Err(e) => {
            error!("RTL8168D: TX port create failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };
    let (rx_write, rx_read) = match channel_create(65536) {
        Ok(h) => h,
        Err(e) => {
            error!("RTL8168D: RX port create failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    // Publish service to VFS
    use abi::syscall::vfs_flags::{O_CREAT, O_RDWR};
    use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};

    let _ = vfs_mkdir("/services/net");
    let _ = vfs_mkdir("/services/net/rtl8168");

    if let Ok(fd) = vfs_open("/services/net/rtl8168/tx", O_CREAT | O_RDWR) {
        let _ = vfs_write(fd, alloc::format!("{}", tx_write).as_bytes());
        let _ = vfs_close(fd);
    }
    if let Ok(fd) = vfs_open("/services/net/rtl8168/rx", O_CREAT | O_RDWR) {
        let _ = vfs_write(fd, alloc::format!("{}", rx_read).as_bytes());
        let _ = vfs_close(fd);
    }
    if let Ok(fd) = vfs_open("/services/net/rtl8168/mac", O_CREAT | O_RDWR) {
        let mac_packed = (mac[0] as u64)
            | ((mac[1] as u64) << 8)
            | ((mac[2] as u64) << 16)
            | ((mac[3] as u64) << 24)
            | ((mac[4] as u64) << 32)
            | ((mac[5] as u64) << 40);
        let _ = vfs_write(fd, alloc::format!("0x{:x}", mac_packed).as_bytes());
        let _ = vfs_close(fd);
    }

    info!(
        "RTL8168D: service ready (tx_port={} rx_port={})",
        tx_write, rx_read
    );

    let mut tx_msg_buf = [0u8; 2048];
    loop {
        // Opportunistically drain TX requests.
        match channel_recv(tx_read, &mut tx_msg_buf) {
            Ok(n) if n > 0 => {
                if let Some(msg) = NetDriverMsg::decode(&tx_msg_buf[..n]) {
                    match msg.msg_type {
                        MSG_FRAME_TX => {
                            if let Err(e) = driver.tx(msg.payload) {
                                warn!("RTL8168D: TX failed: {}", e);
                            }
                        }
                        MSG_MAC_REQ => {
                            let m = NetDriverMsg::new(MSG_MAC_RESP, &mac);
                            let _ = channel_send(rx_write, &m.encode());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        // Interrupt-driven path (falls back to short sleep if MSI isn't available).
        driver.wait_for_irq();
        let isr = driver.consume_interrupts();
        if (isr & (1 << 5)) != 0 {
            // Future link status reporting
        }

        if (isr & ((1 << 0) | (1 << 1) | (1 << 4) | (1 << 6))) != 0 || !driver.irq_enabled() {
            while let Some(frame) = driver.poll_rx() {
                let msg = NetDriverMsg::new(MSG_FRAME_RX, frame);
                if let Err(e) = channel_send(rx_write, &msg.encode()) {
                    warn!("RTL8168D: RX forward failed: {:?}", e);
                    break;
                }
            }
        }
    }
}
