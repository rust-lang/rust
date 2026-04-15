#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::task::{ManagedTask, TaskKind};
use abi::display_driver_protocol::{FbInfoPayload, FB_INFO_PAYLOAD_SIZE};
use abi::ids::HandleId;
use abi::schema::{keys, kinds};
use abi::syscall::vfs_flags::O_RDONLY;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;
use stem::abi::driver_ctx::DriverCtx;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};
use stem::syscall::{channel_create, ChannelHandle};
use stem::{debug, info, warn};

fn file_exists(path: &str) -> bool {
    match vfs_open(path, O_RDONLY) {
        Ok(fd) => {
            let _ = vfs_close(fd);
            true
        }
        Err(_) => false,
    }
}

fn read_trimmed_text(path: &str) -> Option<alloc::string::String> {
    let fd = vfs_open(path, O_RDONLY).ok()?;
    let mut buf = [0u8; 256];
    let n = vfs_read(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);
    if n == 0 {
        return None;
    }

    let s = core::str::from_utf8(&buf[..n]).ok()?.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

pub fn select_serial_shell() -> alloc::string::String {
    // Prefer runtime override, then system default, then known built-in fallback.
    for cfg in ["/run/sprout/shell", "/etc/default/shell"] {
        if let Some(candidate) = read_trimmed_text(cfg) {
            if file_exists(&candidate) {
                return candidate;
            }
            warn!(
                "SPROUT: Ignoring shell override '{}' from {} (missing binary)",
                candidate, cfg
            );
        }
    }

    for candidate in ["/bin/sh", "/bin/smallsh"] {
        if file_exists(candidate) {
            return candidate.to_string();
        }
    }

    "/bin/sh".to_string()
}

fn ensure_session_roots() {
    use stem::syscall::vfs::vfs_mkdir;
    let _ = vfs_mkdir("/session");
    let _ = vfs_mkdir("/session/seat0");
    let _ = vfs_mkdir("/session/seat0/keyboard");
    let _ = vfs_mkdir("/session/seat0/pointer");
    let _ = vfs_mkdir("/session/display");
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayHandles {
    pub drv_req_write: ChannelHandle,
    pub drv_resp_read: ChannelHandle,
    pub bs_id: u32,
    /// Which display backend was selected
    pub backend_name: &'static str,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
}

fn find_sys_device(class_prefix: &str) -> Option<alloc::string::String> {
    let fd = match vfs_open("/sys/devices", O_RDONLY) {
        Ok(fd) => fd,
        Err(_) => return None,
    };

    let mut buf = [0u8; 4096];
    let n = match stem::syscall::vfs::vfs_readdir(fd, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            warn!("SPROUT: readdir(/sys/devices) failed: {:?}", e);
            let _ = vfs_close(fd);
            return None;
        }
    };
    let _ = vfs_close(fd);

    debug!("SPROUT: readdir found {} bytes", n);

    let mut offset = 0usize;
    while offset < n {
        if buf[offset] == 0 {
            offset += 1;
            continue;
        }

        let mut end = offset;
        while end < n && buf[end] != 0 {
            end += 1;
        }

        if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
            debug!("SPROUT:   Checking entry at {}: '{}'", offset, name);
            if name.starts_with("pci-") {
                let class_path = alloc::format!("/sys/devices/{}/class", name);
                if let Ok(class_fd) = vfs_open(&class_path, O_RDONLY) {
                    let mut class_buf = [0u8; 16];
                    if let Ok(cn) = vfs_read(class_fd, &mut class_buf) {
                        let class_str = core::str::from_utf8(&class_buf[..cn]).unwrap_or("");
                        debug!(
                            "SPROUT: Checked device {} class='{}'",
                            name,
                            class_str.trim()
                        );
                        if class_str.trim().starts_with(class_prefix) {
                            let _ = vfs_close(class_fd);
                            return Some(alloc::format!("/sys/devices/{}", name));
                        }
                    } else {
                        debug!("SPROUT: Failed to read {}", class_path);
                    }
                    let _ = vfs_close(class_fd);
                } else {
                    debug!("SPROUT: Failed to open {}", class_path);
                }
            }
        }
        offset = end.saturating_add(1);
    }
    None
}

fn find_sys_device_with_vendor(
    class_prefix: &str,
    vendor_prefix: &str,
) -> Option<alloc::string::String> {
    let fd = match vfs_open("/sys/devices", O_RDONLY) {
        Ok(fd) => fd,
        Err(_) => return None,
    };

    let mut buf = [0u8; 4096];
    let n = match stem::syscall::vfs::vfs_readdir(fd, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            debug!("SPROUT: readdir(/sys/devices) failed: {:?}", e);
            let _ = vfs_close(fd);
            return None;
        }
    };
    let _ = vfs_close(fd);

    let mut offset = 0usize;
    while offset < n {
        if buf[offset] == 0 {
            offset += 1;
            continue;
        }

        let mut end = offset;
        while end < n && buf[end] != 0 {
            end += 1;
        }

        if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
            if name.starts_with("pci-") {
                let class_path = alloc::format!("/sys/devices/{}/class", name);
                let vendor_path = alloc::format!("/sys/devices/{}/vendor", name);

                let class_matches = if let Ok(class_fd) = vfs_open(&class_path, O_RDONLY) {
                    let mut class_buf = [0u8; 16];
                    let matched = if let Ok(cn) = vfs_read(class_fd, &mut class_buf) {
                        let class_str = core::str::from_utf8(&class_buf[..cn]).unwrap_or("");
                        class_str.trim().starts_with(class_prefix)
                    } else {
                        false
                    };
                    let _ = vfs_close(class_fd);
                    matched
                } else {
                    false
                };

                if !class_matches {
                    offset = end.saturating_add(1);
                    continue;
                }

                let vendor_matches = if let Ok(vendor_fd) = vfs_open(&vendor_path, O_RDONLY) {
                    let mut vendor_buf = [0u8; 16];
                    let matched = if let Ok(vn) = vfs_read(vendor_fd, &mut vendor_buf) {
                        let vendor_str = core::str::from_utf8(&vendor_buf[..vn]).unwrap_or("");
                        vendor_str.trim().starts_with(vendor_prefix)
                    } else {
                        false
                    };
                    let _ = vfs_close(vendor_fd);
                    matched
                } else {
                    false
                };

                if vendor_matches {
                    return Some(alloc::format!("/sys/devices/{}", name));
                }
            }
        }

        offset = end.saturating_add(1);
    }

    None
}

fn has_sys_device(class_prefix: &str) -> bool {
    find_sys_device(class_prefix).is_some()
}

fn probe_bootfb_vfs() -> Option<(u32, u32, u32, u32)> {
    let fd = match vfs_open("/dev/fb0", O_RDONLY) {
        Ok(fd) => fd,
        Err(e) => {
            debug!("SPROUT: open(/dev/fb0) failed: {:?}", e);
            return None;
        }
    };
    let mut payload = FbInfoPayload {
        device_handle: 0,
        width: 0,
        height: 0,
        stride: 0,
        bpp: 32,
        format: 0,
        _reserved: 0,
    };
    let slice = unsafe {
        core::slice::from_raw_parts_mut(&mut payload as *mut _ as *mut u8, FB_INFO_PAYLOAD_SIZE)
    };
    let n = match vfs_read(fd, slice) {
        Ok(n) => n,
        Err(e) => {
            let _ = vfs_close(fd);
            debug!("SPROUT: read(/dev/fb0) failed: {:?}", e);
            return None;
        }
    };
    let _ = vfs_close(fd);
    if n < FB_INFO_PAYLOAD_SIZE || payload.width == 0 || payload.height == 0 || payload.stride == 0
    {
        debug!(
            "SPROUT: /dev/fb0 payload invalid: n={} width={} height={} stride={} format={}",
            n, payload.width, payload.height, payload.stride, payload.format
        );
        return None;
    }
    Some((
        payload.width,
        payload.height,
        payload.stride,
        payload.format,
    ))
}

// Storage, Network, and Audio are now handled by devd

pub fn setup_display_pipeline(
    shared_tasks: Arc<Mutex<Vec<ManagedTask>>>,
    supervisor_port: stem::syscall::ChannelHandle,
    bind_instance_id: u64,
) -> Option<DisplayHandles> {
    debug!(
        "SPROUT: setup_display_pipeline start (bind_id={})",
        bind_instance_id
    );

    let mut display_width = 0u32;
    let mut display_height = 0u32;
    let mut display_stride = 0u32;
    let mut display_format = 0u32;
    let mut driver_name: Option<&'static str> = None;
    let mut backend_name: &'static str = "unknown";

    // ThingOS-style BootFB probe: if /dev/fb0 exists, trust that as the canonical display.
    if let Some((w, h, stride, format)) = probe_bootfb_vfs() {
        display_width = w;
        display_height = h;
        display_stride = stride;
        display_format = format;
        driver_name = Some("/bin/display_bootfb");
        backend_name = "BootFB";
        debug!(
            "SPROUT: Using /dev/fb0 boot framebuffer ({}x{} stride={})",
            display_width, display_height, display_stride
        );
    }
    // virtio-gpu: class 0x030000, vendor 0x1af4
    if driver_name.is_none() && has_sys_device("0x0300") {
        display_stride = display_width * 4;
        display_format = 1;
        driver_name = Some("/bin/display_virtio_gpu");
        backend_name = "VirtIO-GPU";
        debug!(
            "SPROUT: Using VirtIO GPU at {}x{}",
            display_width, display_height
        );
    }

    // Fallback to BootFB already handled by probe_bootfb_vfs() at start of function

    // Fallback to display_fake if no other display found (ensures Bloom launches)
    if driver_name.is_none() && backend_name != "BootFB" {
        debug!("SPROUT: No display device found! Using display_fake (headless mode)");
        display_width = 1024;
        display_height = 768;
        display_stride = 1024 * 4;
        display_format = 1; // BGRA8888
        driver_name = Some("/bin/display_fake");
        backend_name = "Fake";
    }

    if display_width == 0 || display_height == 0 || display_stride == 0 {
        debug!("SPROUT: Invalid display geometry, skipping display pipeline");
        return None;
    }

    debug!(
        "SPROUT: Display backend selected: {} (Driver: {:?}, Geometry: {}x{} stride={})",
        backend_name, driver_name, display_width, display_height, display_stride
    );

    let size = (display_height as usize) * (display_stride as usize);
    let bs_id = match stem::syscall::memfd_create("display.buffer", size) {
        Ok(fd) => fd,
        Err(e) => {
            debug!("SPROUT: memfd_create failed: {:?}", e);
            return None;
        }
    };

    let _ = bs_id; // Metadata is now passed via bloom bootstrap bytespace

    let mut drv_req_write = 0;
    let mut drv_resp_read = 0;

    if let Some(driver_name) = driver_name {
        let drv_req = match channel_create(4096) {
            Ok(handles) => handles,
            Err(e) => {
                debug!("SPROUT: drv_req channel_create failed: {:?}", e);
                return None;
            }
        };
        let drv_resp = match channel_create(4096) {
            Ok(handles) => handles,
            Err(e) => {
                debug!("SPROUT: drv_resp channel_create failed: {:?}", e);
                return None;
            }
        };

        drv_req_write = drv_req.0;
        drv_resp_read = drv_resp.1;

        let boot_size = 4096;
        let boot_fd = stem::syscall::memfd_create("driver.boot", boot_size).unwrap_or(0);
        if boot_fd != 0 {
            use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
            let req = VmMapReq {
                addr_hint: 0,
                len: boot_size,
                prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
                flags: VmMapFlags::empty(),
                backing: VmBacking::File {
                    fd: boot_fd,
                    offset: 0,
                },
            };
            if let Ok(resp) = stem::syscall::vm_map(&req) {
                let ptr = resp.addr;
                let slice =
                    unsafe { core::slice::from_raw_parts_mut(ptr as *mut u32, boot_size / 4) };

                slice[0] = drv_req.1 as u32; // Read end of req channel
                slice[1] = drv_resp.0 as u32; // Write end of resp channel
                slice[2] = if supervisor_port != 0 {
                    supervisor_port
                } else {
                    drv_resp.0 as u32
                };

                let id_low = (bind_instance_id & 0xFFFF_FFFF) as u32;
                let id_high = (bind_instance_id >> 32) as u32;
                slice[3] = id_low;
                slice[4] = id_high;

                debug!("SPROUT: Bootstrapping driver {} via memfd {}: req_r={}, resp_w={}, svc={}, id={}", 
                    driver_name, boot_fd, slice[0], slice[1], slice[2], bind_instance_id);
            }
        }

        let argv_str = if boot_fd > 0 {
            alloc::format!("{}", boot_fd)
        } else {
            "none".to_string()
        };

        let spawn_res = stem::syscall::spawn_process_ex(
            driver_name,
            &[driver_name.as_bytes(), argv_str.as_bytes()],
            &alloc::collections::BTreeMap::new(),
            stem::abi::types::stdio_mode::INHERIT,
            stem::abi::types::stdio_mode::INHERIT,
            stem::abi::types::stdio_mode::INHERIT,
            boot_fd as u64, // boot_arg
            &[drv_req.1 as u64, drv_resp.0 as u64],
        );

        if let Ok(resp) = spawn_res {
            let pid = resp.child_tid;
            debug!(
                "SPROUT: Spawned display driver '{}' (PID={})",
                driver_name, pid
            );
            let _ = stem::thread::set_priority(pid, 3);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: driver_name.to_string(),
                kind: TaskKind::Driver("dev.display".to_string()),
                module_path: driver_name.to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: boot_fd as usize,
                bind_instance_id: bind_instance_id,
                drv_req_write: drv_req.0,
                drv_resp_read: drv_resp.1,
                boot_req_read: drv_req.1,
                boot_resp_write: drv_resp.0,
            });
        }

        // Legacy graph nodes removed. display-os protocol will carry backend specifics.
    }

    Some(DisplayHandles {
        drv_req_write,
        drv_resp_read,
        bs_id,
        backend_name,
        width: display_width,
        height: display_height,
        stride: display_stride,
        format: display_format,
    })
}

pub fn setup_terminal(
    shared_tasks: Arc<Mutex<Vec<ManagedTask>>>,
    display: Option<DisplayHandles>,
    _input: InputHandles,
) {
    debug!("SPROUT: Setting up Terminal...");
    ensure_session_roots();

    let Some(display) = display else {
        debug!("SPROUT: Cannot setup terminal without display!");
        return;
    };

    let boot_size = 4096;
    let boot_fd = stem::syscall::memfd_create("terminal.boot", boot_size).unwrap_or(0);

    if boot_fd != 0 {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: boot_size,
            prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: boot_fd,
                offset: 0,
            },
        };
        if let Ok(resp) = stem::syscall::vm_map(&req) {
            let ptr = resp.addr;
            let slice = unsafe { core::slice::from_raw_parts_mut(ptr as *mut u32, boot_size / 4) };
            slice[0] = 0xB100AA01; // Magic
            slice[1] = display.drv_req_write as u32;
            slice[2] = display.drv_resp_read as u32;
            slice[4] = display.bs_id;
            debug!(
                "SPROUT: Bootstrapping terminal via memfd {}: req_w={}, resp_r={}, bs_id={}",
                boot_fd, slice[1], slice[2], slice[4]
            );
        }
    }

    let term_arg = boot_fd as u32;

    match stem::syscall::spawn_process("/bin/terminal", term_arg as usize) {
        Ok(pid) => {
            debug!("SPROUT: Spawned terminal (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "terminal".to_string(),
                kind: TaskKind::App,
                module_path: "/bin/terminal".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: term_arg as usize,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to spawn terminal: {:?}", e);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InputHandles {
    pub bloom_evt_read: ChannelHandle,
    pub evt_input_echo_read: ChannelHandle,
}

pub fn setup_input_broker(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) -> InputHandles {
    debug!("SPROUT: Setting up input pipeline (keyboard + mouse)...");

    // Create kbd_raw port (ps2_kbd -> bristle)
    let kbd_raw = match stem::syscall::channel_create(4096) {
        Ok((write_h, read_h)) => {
            debug!("SPROUT: Created kbd_raw port (w={}, r={})", write_h, read_h);
            (write_h, read_h)
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to create kbd_raw port: {:?}", e);
            return InputHandles {
                bloom_evt_read: 0,
                evt_input_echo_read: 0,
            };
        }
    };

    // Create mouse_raw port (ps2_mouse -> bristle)
    let mouse_raw = match stem::syscall::channel_create(4096) {
        Ok((write_h, read_h)) => {
            debug!(
                "SPROUT: Created mouse_raw port (w={}, r={})",
                write_h, read_h
            );
            (write_h, read_h)
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to create mouse_raw port: {:?}", e);
            return InputHandles {
                bloom_evt_read: 0,
                evt_input_echo_read: 0,
            };
        }
    };

    // Dedicated Bristle -> Bloom input channel plus optional input echo tap.
    let bloom_evt = match stem::syscall::channel_create(4096) {
        Ok(h) => h,
        Err(e) => {
            stem::error!("SPROUT: Failed to create bloom_evt: {:?}", e);
            return InputHandles {
                bloom_evt_read: 0,
                evt_input_echo_read: 0,
            };
        }
    };

    let evt_input_echo = match stem::syscall::channel_create(4096) {
        Ok(h) => h,
        Err(e) => {
            stem::error!("SPROUT: Failed to create evt_input_echo: {:?}", e);
            return InputHandles {
                bloom_evt_read: 0,
                evt_input_echo_read: 0,
            };
        }
    };

    // Spawn ps2_kbd with raw write handle
    match stem::syscall::spawn_process("/bin/ps2_kbd", kbd_raw.0 as usize) {
        Ok(pid) => {
            debug!("SPROUT: Spawned ps2_kbd (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "/ps2_kbd".to_string(),
                kind: TaskKind::Driver("dev.input.ps2.kbd".to_string()),
                module_path: "/bin/ps2_kbd".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: kbd_raw.0 as usize,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to spawn ps2_kbd: {:?}", e);
        }
    }

    // Spawn ps2_mouse with raw write handle
    match stem::syscall::spawn_process("/bin/ps2_mouse", mouse_raw.0 as usize) {
        Ok(pid) => {
            debug!("SPROUT: Spawned ps2_mouse (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "ps2_mouse".to_string(),
                kind: TaskKind::Driver("dev.input.ps2.mouse".to_string()),
                module_path: "/bin/ps2_mouse".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: mouse_raw.0 as usize,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to spawn ps2_mouse: {:?}", e);
        }
    }

    // Layout: kbd_raw_read[63:48] | mouse_raw_read[47:32] | evt_input_echo_write[15:0]
    let bristle_arg = ((kbd_raw.1 as u64) << 48)
        | ((mouse_raw.1 as u64) << 32)
        | ((bloom_evt.0 as u64) << 16)
        | (evt_input_echo.0 as u64);

    match stem::syscall::spawn_process("/bin/bristle", bristle_arg as usize) {
        Ok(pid) => {
            debug!("SPROUT: Spawned bristle (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "bristle".to_string(),
                kind: TaskKind::App,
                module_path: "/bin/bristle".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: bristle_arg as usize,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            stem::error!("SPROUT: Failed to spawn bristle: {:?}", e);
        }
    }

    debug!("SPROUT: Input broker ready (keyboard + mouse)");
    stem::sleep_ms(100);
    InputHandles {
        bloom_evt_read: bloom_evt.1,
        evt_input_echo_read: evt_input_echo.1,
    }
}

/// Set up network pipeline - spawn virtio_netd (driver) then netd (stack)
// Network and Audio are now handled by devd

fn spawn_netd(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    debug!("SPROUT: spawn_netd start");
    match stem::syscall::spawn_process("/bin/netd", 0) {
        Ok(pid) => {
            debug!("SPROUT: Spawned netd (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "netd".to_string(),
                kind: TaskKind::Service("svc.net".to_string()),
                module_path: "/bin/netd".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn netd: {:?}", e);
        }
    }
}

pub fn setup_network_apps(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    debug!("SPROUT: Setting up network apps...");

    match stem::syscall::spawn_process("/bin/nectar", 0) {
        Ok(pid) => {
            debug!("SPROUT: Spawned nectar (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "nectar".to_string(),
                kind: TaskKind::Service("svc.nectar".to_string()),
                module_path: "/bin/nectar".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn nectar: {:?}", e);
        }
    }

    match stem::syscall::spawn_process("/bin/fetchd", 0) {
        Ok(pid) => {
            debug!("SPROUT: Spawned fetchd (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "fetchd".to_string(),
                kind: TaskKind::App,
                module_path: "/bin/fetchd".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn fetchd: {:?}", e);
        }
    }
}

pub fn setup_taskman_service(_shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    // Taskman removed
}

pub fn setup_ui_services(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    spawn_ui_service(shared_tasks.clone(), "/bin/flytrap", "svc.flytrap", 2);
    spawn_ui_service(shared_tasks, "/bin/blossom", "svc.blossom", 2);
}

pub fn setup_blossom_service(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    spawn_ui_service(shared_tasks, "/bin/blossom", "svc.blossom", 2);
}

pub fn setup_font_service(_shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    // Font handling is integrated into Bloom directly
}

pub fn setup_flytrap_service(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    spawn_ui_service(shared_tasks, "/bin/flytrap", "svc.flytrap", 2);
}

fn spawn_ui_service(
    shared_tasks: Arc<Mutex<Vec<ManagedTask>>>,
    name: &str,
    service: &str,
    priority: usize,
) {
    {
        let tasks = shared_tasks.lock();
        if tasks.iter().any(|t| t.name == name && t.pid.is_some()) {
            return;
        }
    }

    match stem::syscall::spawn_process(name, 0) {
        Ok(pid) => {
            debug!("SPROUT: Spawned {} (PID={})", &name[1..], pid);
            let _ = stem::thread::set_priority(pid, priority);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: name.to_string(),
                kind: TaskKind::Service(service.to_string()),
                module_path: name.to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn {}: {:?}", &name[1..], e);
        }
    }
}

/// Maximum time to wait for the audio VFS node to appear after spawning the driver.
const AUDIO_DEVICE_TIMEOUT_NS: u64 = 5_000_000_000; // 5 seconds

/// Poll interval while waiting for `/dev/audio/card0/out0` to appear.
const AUDIO_POLL_INTERVAL_MS: u64 = 100;

/// Probe for an audio device, spawn the right driver, wait for the VFS node
/// to appear, then launch the beeper to play the start-up chime.
///
/// Detection order (first match wins):
///   1. VirtIO sound — PCI class 0x0401xx **and** vendor 0x1af4
///   2. Intel HDA    — PCI class 0x0403xx
pub fn setup_audio_stack(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    debug!("SPROUT: setup_audio_stack: probing for audio hardware...");

    // Pick the driver binary that matches the detected hardware.
    let driver: Option<&'static str> =
        if find_sys_device_with_vendor("0x0401", "0x1af4").is_some() {
            info!("SPROUT: Found VirtIO sound device");
            Some("/bin/virtio_sound")
        } else if find_sys_device("0x0403").is_some() {
            info!("SPROUT: Found HDA controller (class 0x0403)");
            Some("/bin/hdaudio")
        } else {
            info!("SPROUT: No audio hardware detected; skipping audio stack");
            None
        };

    let driver_path = match driver {
        Some(p) => p,
        None => return,
    };

    // Spawn the audio driver.
    match stem::syscall::spawn_process(driver_path, 0) {
        Ok(pid) => {
            info!("SPROUT: Spawned audio driver '{}' (PID={})", driver_path, pid);
            let _ = stem::thread::set_priority(pid, 3);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: driver_path.to_string(),
                kind: TaskKind::Driver("dev.audio".to_string()),
                module_path: driver_path.to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn audio driver '{}': {:?}", driver_path, e);
            return;
        }
    }

    // Wait for the audio output stream to appear in the VFS.
    let deadline_ns = stem::monotonic_ns() + AUDIO_DEVICE_TIMEOUT_NS;
    loop {
        use abi::syscall::vfs_flags::O_RDWR;
        match stem::syscall::vfs::vfs_open("/dev/audio/card0/out0", O_RDWR) {
            Ok(fd) => {
                let _ = stem::syscall::vfs::vfs_close(fd);
                info!("SPROUT: /dev/audio/card0/out0 is ready");
                break;
            }
            Err(_) => {
                if stem::monotonic_ns() >= deadline_ns {
                    warn!("SPROUT: Timeout waiting for /dev/audio/card0/out0; beeper will not start");
                    return;
                }
                stem::time::sleep_ms(AUDIO_POLL_INTERVAL_MS);
            }
        }
    }

    // Spawn the beeper to play the start-up chime.
    match stem::syscall::spawn_process("/bin/beeper", 0) {
        Ok(pid) => {
            info!("SPROUT: Spawned beeper (PID={})", pid);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "beeper".to_string(),
                kind: TaskKind::App,
                module_path: "/bin/beeper".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn beeper: {:?}", e);
        }
    }
}

pub fn setup_graphics_stack(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    debug!("SPROUT: Setting up Graphics Stack (Bloom + fontd)...");

    // 1. Setup fontd
    //
    // fontd requires a paired channel: a request channel for receiving client
    // requests and a reply channel for sending responses back.  We pass both
    // handles to fontd as a single packed argument:
    //   arg0 = (reply_write_h << 16) | req_read_h
    //
    // Clients (e.g. bloom) receive:
    //   req_write_h — to send requests to fontd
    //   rep_read_h  — to receive replies from fontd
    let req_chan = channel_create(4096).expect("Failed to create fontd request channel");
    let rep_chan = channel_create(4096).expect("Failed to create fontd reply channel");
    // req_chan: (req_write_h, req_read_h) — clients write, fontd reads
    // rep_chan: (rep_write_h, rep_read_h) — fontd writes, clients read
    let fontd_arg = ((rep_chan.0 as usize) << 16) | (req_chan.1 as usize);

    match stem::syscall::spawn_process("/bin/fontd", fontd_arg) {
        Ok(pid) => {
            debug!("SPROUT: Spawned fontd (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 2);
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "fontd".to_string(),
                kind: TaskKind::Service("svc.font".to_string()),
                module_path: "/bin/fontd".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: fontd_arg,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn fontd: {:?}", e);
        }
    }

    // 2. Setup Bloom (Compositor)
    match stem::syscall::spawn_process("/bin/bloom", 0) {
        Ok(pid) => {
            debug!("SPROUT: Spawned bloom (PID={})", pid);
            let _ = stem::thread::set_priority(pid, 3); // High priority for compositor
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "bloom".to_string(),
                kind: TaskKind::App,
                module_path: "/bin/bloom".to_string(),
                pid: Some(pid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn bloom: {:?}", e);
        }
    }
}

pub fn setup_serial_shell(shared_tasks: Arc<Mutex<Vec<ManagedTask>>>) {
    debug!("SPROUT: Setting up serial shell on /dev/console...");
    let shell_path = select_serial_shell();

    let console_fd =
        match stem::syscall::vfs::vfs_open("/dev/console", abi::syscall::vfs_flags::O_RDWR) {
            Ok(fd) => fd,
            Err(e) => {
                warn!("SPROUT: Failed to open /dev/console for shell: {:?}", e);
                return;
            }
        };

    match stem::syscall::spawn_process_ex(
        &shell_path,
        &[shell_path.as_bytes()],
        &alloc::collections::BTreeMap::new(),
        abi::types::stdio_mode::INHERIT, // stdin
        abi::types::stdio_mode::INHERIT, // stdout
        abi::types::stdio_mode::INHERIT, // stderr
        0,
        &[],
    ) {
        Ok(resp) => {
            debug!(
                "SPROUT: Spawned serial shell '{}' (PID={})",
                shell_path, resp.child_tid
            );
            let mut tasks = shared_tasks.lock();
            tasks.push(ManagedTask {
                name: "shell".to_string(),
                kind: TaskKind::App,
                module_path: shell_path,
                pid: Some(resp.child_tid),
                restarts: 0,
                spawn_arg: 0,
                bind_instance_id: 0,
                drv_req_write: 0,
                drv_resp_read: 0,
                boot_req_read: 0,
                boot_resp_write: 0,
            });
        }
        Err(e) => {
            warn!("SPROUT: Failed to spawn serial shell: {:?}", e);
            let _ = vfs_close(console_fd);
        }
    }
}
