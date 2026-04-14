#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::device::PCI_IRQ_MODE_MSIX;
use core::ptr::write_volatile;
use core::sync::atomic::{AtomicUsize, Ordering};
use stem::abi::module_manifest::{MANIFEST_MAGIC, ManifestHeader, ModuleKind};
use stem::device::device_enable_msi;
use stem::syscall::{device_alloc_dma, device_dma_phys, device_irq_subscribe, device_irq_wait};
use stem::thread;
use stem::{error, info, warn};

use virtio_gpu::{Rect, VirtioGpu};

static IRQ_HANDLE: AtomicUsize = AtomicUsize::new(0);
const DEMO_RESOURCE_ID: u32 = 1;

#[unsafe(link_section = ".thing_manifest")]
#[unsafe(no_mangle)]
#[used]
pub static MANIFEST: ManifestHeader = ManifestHeader {
    magic: MANIFEST_MAGIC,
    kind: ModuleKind::Driver,
    device_kind: *b"dev.display.Gpu\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
    version: 1,
    _reserved: 0,
};

#[stem::main]
fn main(boot_fd: usize) -> ! {
    info!("VIRTIO_GPU: Starting driver, boot_fd={}", boot_fd);

    // 1. Get device path from bootstrap memfd
    let mut path_buf = [0u8; 128];
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
            b"/sys/devices/pci-00:02.0" // Fallback
        }
    } else {
        b"/sys/devices/pci-00:02.0"
    };
    let path_str = core::str::from_utf8(path).unwrap_or("");

    info!("VIRTIO_GPU: Using path={}", path_str);

    let mut drv_req_read = 0;
    let mut drv_resp_write = 0;
    let mut supervisor_port = 0;
    let mut bind_instance_id = 0u64;

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
            drv_req_read = slice[0];
            drv_resp_write = slice[1];
            supervisor_port = slice[2];
            let id_low = slice[3] as u64;
            let id_high = slice[4] as u64;
            bind_instance_id = id_low | (id_high << 32);
            info!(
                "VIRTIO_GPU: Bootstrap handles: req_read={}, resp_write={}, svc={}, id={}",
                drv_req_read, drv_resp_write, supervisor_port, bind_instance_id
            );
        }
    }

    // 2. Initialize Hardware
    let mut gpu = match VirtioGpu::new(path_str) {
        Ok(g) => g,
        Err(e) => {
            error!("VIRTIO_GPU: Failed to init driver: {:?}", e);
            stem::syscall::exit(1);
        }
    };

    // Initialize virtio
    if let Err(e) = gpu.init_virtio() {
        error!("VIRTIO_GPU: Virtio init failed: {}", e);
        stem::syscall::exit(1);
    }

    // Enable MSI-X if available
    match device_enable_msi(gpu.claim_handle(), true) {
        Ok(resp) => {
            info!(
                "VIRTIO_GPU: IRQ mode {} vector=0x{:02x}",
                resp.irq_mode, resp.vector
            );
            if resp.irq_mode == PCI_IRQ_MODE_MSIX {
                configure_msix(&gpu);
            }
            if let Err(e) = device_irq_subscribe(gpu.claim_handle(), 0) {
                warn!("VIRTIO_GPU: device IRQ subscribe failed: {:?}", e);
            } else {
                IRQ_HANDLE.store(gpu.claim_handle(), Ordering::Release);
                let _ = thread::spawn(irq_thread);
            }
        }
        Err(e) => warn!("VIRTIO_GPU: MSI enable failed: {:?}", e),
    }

    // Setup the display pipeline
    if let Err(e) = setup_display(&mut gpu) {
        error!("VIRTIO_GPU: Display setup failed: {}", e);
        stem::syscall::exit(1);
    }

    // 3. Register as VFS Provider via Sovereign Handshake
    use abi::vfs_rpc::VFS_RPC_MAX_REQ;
    let (vfs_write, vfs_read) =
        stem::syscall::channel_create(VFS_RPC_MAX_REQ * 8).expect("Failed to create VFS channel");

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
            info!(
                "VIRTIO_GPU: Sending MSG_BIND_READY handshake (ID: {})...",
                bind_instance_id
            );
            // Bundle the VFS provider handle and the BIND_READY notification atomically.
            let _ = stem::syscall::channel::channel_send_msg(
                drv_resp_write,
                &buf[..total_len],
                &[vfs_write],
            );
        }
    }

    // Wait for MSG_BIND_ASSIGNED
    let mut wait_buf = [0u8; 512];
    loop {
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
                        info!(
                            "VIRTIO_GPU: Sovereign registration COMPLETE. Assigned: {}",
                            path
                        );
                        break;
                    }
                }
            }
        }
        stem::syscall::yield_now();
    }

    info!("VIRTIO_GPU: Driver initialized, entering demo loop");

    // Get framebuffer for demo
    let framebuffer = match create_demo_framebuffer(&mut gpu) {
        Ok(fb) => fb,
        Err(e) => {
            error!("VIRTIO_GPU: Failed to create demo framebuffer: {}", e);
            stem::syscall::exit(1);
        }
    };

    // Simple animation loop to prove it works
    let mut frame = 0u32;
    let (w, h) = gpu.get_dimensions();
    loop {
        // Draw animated pattern
        let fb = framebuffer as *mut u32;

        for y in 0..h as usize {
            for x in 0..w as usize {
                let offset = ((x + frame as usize) % 100) * 2;
                let color = if (y + offset) % 40 < 20 {
                    0x00FF0000 // Red in BGRA
                } else {
                    0x000000FF // Blue in BGRA
                };
                unsafe { write_volatile(fb.add(y * w as usize + x), color) };
            }
        }

        // Flush to display
        let full_rect = Rect { x: 0, y: 0, w, h };
        let _ = gpu.present_rect(DEMO_RESOURCE_ID, full_rect);

        if frame % 60 == 0 {
            info!("VIRTIO_GPU: Frame {}", frame);
        }

        frame = frame.wrapping_add(1);
        stem::syscall::sleep_ms(16); // ~60fps
    }
}

fn setup_display(gpu: &mut VirtioGpu) -> Result<(), &'static str> {
    // Create GPU resource
    gpu.create_resource_2d(DEMO_RESOURCE_ID)?;

    info!("VIRTIO_GPU: Display pipeline ready!");
    Ok(())
}

fn create_demo_framebuffer(gpu: &mut VirtioGpu) -> Result<u64, &'static str> {
    let (width, height) = gpu.get_dimensions();
    let fb_size = (width * height * 4) as usize;
    let pages = (fb_size + 4095) / 4096;

    let framebuffer =
        device_alloc_dma(gpu.claim_handle(), pages).map_err(|_| "Failed to alloc framebuffer")?;
    let fb_phys = device_dma_phys(framebuffer).map_err(|_| "Failed to get fb phys")?;

    info!(
        "VIRTIO_GPU: Framebuffer {}x{} @ virt=0x{:x} phys=0x{:x}",
        width, height, framebuffer, fb_phys
    );

    // Clear framebuffer to a visible color (bright green)
    let fb = framebuffer as *mut u32;
    for i in 0..(width * height) as usize {
        unsafe { write_volatile(fb.add(i), 0x0000FF00) }; // Green in BGRA
    }

    // Attach framebuffer memory to resource
    gpu.attach_backing(DEMO_RESOURCE_ID, fb_phys, fb_size, width * 4)?;

    // Set this resource as scanout 0
    gpu.set_scanout(DEMO_RESOURCE_ID, width, height)?;

    // Initial transfer + flush
    let full_rect = Rect {
        x: 0,
        y: 0,
        w: width,
        h: height,
    };
    gpu.present_rect(DEMO_RESOURCE_ID, full_rect)?;

    Ok(framebuffer)
}

fn configure_msix(gpu: &VirtioGpu) {
    // MSI-X configuration is handled internally by the virtio device
    // The GPU struct doesn't expose the raw MMIO regions for security
    // This function is kept as a placeholder for future enhancements
    let _ = gpu;
}

extern "C" fn irq_thread() -> ! {
    let claim_handle = IRQ_HANDLE.load(Ordering::Acquire);
    loop {
        match device_irq_wait(claim_handle, 0) {
            Ok(count) => info!("VIRTIO_GPU: IRQ fired ({})", count),
            Err(e) => {
                warn!("VIRTIO_GPU: IRQ wait error {:?}", e);
                stem::yield_now();
            }
        }
    }
}
