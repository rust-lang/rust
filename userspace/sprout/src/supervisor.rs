//! Supervisor service for Thing-OS (sprout)
//!
//! Orchestrates the system boot sequence:
//! 1. VFS Namespace management (via memfds and mounts).
//! 2. Sovereign registration handshake (receiving handles from drivers).
//! 3. Graphics stack bring-up (coordinating display + fonts + bloom).
//! 4. Monitoring device arrivals and spawning dependent services.
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

// Modules are now declared in main.rs
use alloc::sync::Arc;
use alloc::vec::Vec;

use abi::display_driver_protocol;
use abi::supervisor_protocol::{
    self, MSG_BIND_ASSIGNED, MSG_BIND_FAILED, MSG_BIND_READY, MSG_SERVICE_EXITING,
    MSG_SERVICE_READY, classes,
};
use spin::Mutex;
use stem::syscall::{ChannelHandle, channel_create, channel_send_all, vfs_mount};
use stem::{debug, error, info, warn};

use crate::ledger::DeviceLedger;
use crate::pipelines::{
    DisplayHandles, setup_audio_stack, setup_display_pipeline, setup_graphics_stack,
    setup_input_broker, setup_serial_shell,
};
use crate::task::{ManagedTask, TaskKind};

pub struct Supervisor {
    pub tasks: Arc<Mutex<Vec<ManagedTask>>>,
    pub ledger: Arc<Mutex<DeviceLedger>>,
    pub registry_ptr: usize,
}

impl Supervisor {
    pub fn new(registry_ptr: usize) -> Self {
        Self {
            tasks: Arc::new(Mutex::new(Vec::new())),
            ledger: Arc::new(Mutex::new(DeviceLedger::new())),
            registry_ptr,
        }
    }

    pub fn run_forever(&mut self) -> ! {
        stem::debug!("SPROUT: Supervisor session started (Sovereign mode)");

        // Stage 1: Discover boot modules
        self.discover();

        // Stage 2: Create Sovereign Registrar channel
        let (supervisor_write, supervisor_read) =
            channel_create(4096).expect("Failed to create supervisor registrar channel");

        // Stage 3: Launch Serial Shell EARLY on its own processor
        info!("SPROUT: Launching early serial shell...");
        let tasks_cloned = self.tasks.clone();
        let _ = stem::thread::spawn_task(move || {
            setup_serial_shell(tasks_cloned);
        });

        // Stage 4: Fan-out Setup Pipelines in parallel
        info!("SPROUT: Fanning out setup pipelines...");

        let tasks_for_display = self.tasks.clone();
        let _ = stem::thread::spawn_task(move || {
            let _ = setup_display_pipeline(tasks_for_display, 0, 1);
        });

        let tasks_for_graphics = self.tasks.clone();
        let _ = stem::thread::spawn_task(move || {
            setup_graphics_stack(tasks_for_graphics);
        });

        let tasks_for_input = self.tasks.clone();
        let _ = stem::thread::spawn_task(move || {
            setup_input_broker(tasks_for_input);
        });

        let tasks_for_audio = self.tasks.clone();
        let _ = stem::thread::spawn_task(move || {
            setup_audio_stack(tasks_for_audio);
        });

        // Stage 8: Run Readiness Model Verification Test
        info!("SPROUT: Spawning poll_mux verification test...");
        let _ = stem::syscall::spawn_process("/bin/poll_mux", 0);

        // Stage 4: Busy Stage - Wait for Display Driver to register its VFS provider
        // self.wait_for_display();

        loop {
            stem::trace!(
                "SPROUT: --- Supervisor Loop Cycle Start (tasks={}) ---",
                self.tasks.lock().len()
            );
            self.process_registrations();
            self.monitor();
            stem::trace!("SPROUT: --- Supervisor Loop Cycle End ---");
            stem::sleep_ms(100);
        }
    }

    fn wait_for_display(&mut self) {
        stem::debug!("SPROUT: Waiting for display driver registration...");
        let start = stem::monotonic_ns();
        let timeout = 5_000_000_000; // 5 seconds
        let mut step = 0;

        loop {
            self.process_registrations();
            self.monitor();

            if step % 20 == 0 {
                stem::debug!("SPROUT: Still waiting for display (step {})...", step);
            }
            if step % 100 == 0 {
                stem::debug!(
                    "SPROUT: Health check: Loop still running, tasks={}",
                    self.tasks.lock().len()
                );
            }
            step += 1;

            if let Ok(fd) = stem::syscall::vfs::vfs_open(
                "/dev/display/card0",
                stem::abi::syscall::vfs_flags::O_RDONLY,
            ) {
                let _ = stem::syscall::vfs::vfs_close(fd);
                stem::info!("SPROUT: Display card0 detected. Proceeding.");
                break;
            }

            if stem::monotonic_ns() - start > timeout {
                warn!("SPROUT: Timeout waiting for display driver! UI may fail.");
                break;
            }

            stem::sleep_ms(100);
        }
    }

    fn discover(&mut self) {
        // We no longer auto-spawn everything in /bin.
        // We only scan to keep the registry metadata if needed.
        stem::debug!("SPROUT: Discovery loop disabled in favor of devd.");
    }

    fn spawn_devd(&mut self) {
        let (write, read) = match stem::syscall::channel_create(4096) {
            Ok(h) => h,
            Err(_) => return,
        };
        // We could pass the registrar port to devd if it needs to register things,
        // but for now devd just spawns drivers.
        match stem::syscall::spawn_process("/bin/devd", 0) {
            Ok(pid) => {
                info!("SPROUT: Spawned devd (PID={})", pid);
                let mut tasks = self.tasks.lock();
                tasks.push(ManagedTask {
                    name: "devd".to_string(),
                    kind: TaskKind::Service("svc.devd".to_string()),
                    module_path: "/bin/devd".to_string(),
                    pid: Some(pid),
                    restarts: 0,
                    spawn_arg: 0,
                    bind_instance_id: 0,
                    drv_req_write: write,
                    drv_resp_read: read,
                    boot_req_read: 0,
                    boot_resp_write: 0,
                });
            }
            Err(e) => warn!("SPROUT: Failed to spawn devd: {:?}", e),
        }
    }

    pub fn monitor(&mut self) {
        // Collect PIDs to check without holding the lock for the whole operation
        let pids: Vec<(u64, alloc::string::String)> = {
            let tasks = self.tasks.lock();
            tasks.iter().filter_map(|t| t.pid.map(|p| (p as u64, t.name.clone()))).collect()
        };

        if pids.is_empty() {
            return;
        }

        for (pid, name) in pids {
            match stem::syscall::task_poll(pid) {
                Ok((status, code)) => {
                    stem::debug!(
                        "SPROUT: Polling task '{}' (PID {}): status={:?}, code={}",
                        name,
                        pid,
                        status,
                        code
                    );
                    if status == stem::abi::types::TaskStatus::Dead {
                        info!("SPROUT: Task '{}' (PID {}) is Dead (code {})", name, pid, code);

                        // Re-lock to update task state
                        let mut tasks = self.tasks.lock();
                        if let Some(task) = tasks.iter_mut().find(|t| t.pid == Some(pid)) {
                            info!(
                                "SPROUT: Task '{}' (PID {}) died with code {}. Restarting...",
                                task.name, pid, code
                            );
                            task.pid = None;
                            task.restarts += 1;
                        }
                    }
                }
                Err(_) => {
                    let mut tasks = self.tasks.lock();
                    if let Some(task) = tasks.iter_mut().find(|t| t.pid == Some(pid)) {
                        task.pid = None;
                    }
                }
            }
        }

        // Init-style lifecycle: if all supervised children are gone, halt the system.
        let should_shutdown = {
            let tasks = self.tasks.lock();
            !tasks.is_empty() && tasks.iter().all(|t| t.pid.is_none())
        };
        if should_shutdown {
            info!("SPROUT: All supervised tasks have exited. Performing system shutdown...");
            stem::syscall::shutdown();
        }

        // Handle Spawning/Restarting for tasks without PIDs
        let mut tasks = self.tasks.lock();
        for task in tasks.iter_mut() {
            if task.pid.is_none() {
                if task.name == "shell" {
                    let selected = crate::pipelines::select_serial_shell();
                    if task.module_path != selected {
                        info!(
                            "SPROUT: Switching serial shell from '{}' to '{}'",
                            task.module_path, selected
                        );
                        task.module_path = selected;
                    }
                }

                let handles_owned: Vec<u64> =
                    if task.boot_req_read != 0 && task.boot_resp_write != 0 {
                        alloc::vec![task.boot_req_read as u64, task.boot_resp_write as u64]
                    } else {
                        alloc::vec![]
                    };

                let arg_str = alloc::format!("{}", task.spawn_arg);
                let spawn_path = if task.module_path.is_empty() {
                    task.name.as_str()
                } else {
                    task.module_path.as_str()
                };
                let spawn_res = stem::syscall::spawn_process_ex(
                    spawn_path,
                    &[spawn_path.as_bytes(), arg_str.as_bytes()],
                    &alloc::collections::BTreeMap::new(),
                    stem::abi::types::stdio_mode::INHERIT,
                    stem::abi::types::stdio_mode::INHERIT,
                    stem::abi::types::stdio_mode::INHERIT,
                    task.spawn_arg as u64,
                    &handles_owned,
                );

                if let Ok(resp) = spawn_res {
                    task.pid = Some(resp.child_tid);
                    if let TaskKind::Driver(_) = task.kind {
                        let _ = stem::thread::set_priority(resp.child_tid, 3);
                    }
                }
            }
        }
    }

    fn process_registrations(&mut self) {
        // Collect tasks that need polling
        let poll_set: Vec<(u32, u32, alloc::string::String)> = {
            let tasks = self.tasks.lock();
            tasks
                .iter()
                .filter(|t| t.drv_resp_read != 0 && t.pid.is_some())
                .map(|t| (t.drv_resp_read, t.drv_req_write, t.name.clone()))
                .collect()
        };

        for (drv_resp_read, drv_req_write, task_name) in poll_set {
            let mut msg_data = [0u8; 1024];
            let mut msg_fds = [0u32; 1];
            let mut process_count = 0;

            while let Ok((n, n_fds)) =
                stem::syscall::channel::channel_recv_msg(drv_resp_read, &mut msg_data, &mut msg_fds)
            {
                if n == 0 && n_fds == 0 {
                    break;
                }

                process_count += 1;
                if process_count > 32 {
                    stem::warn!(
                        "SPROUT: Throttling registration processing for task '{}'",
                        task_name
                    );
                    break;
                }
                let bundled_fd = if n_fds > 0 { msg_fds[0] } else { 0 };

                if let Some((header, payload)) =
                    abi::display_driver_protocol::parse_message(&msg_data[..n])
                {
                    if header.msg_type == abi::supervisor_protocol::MSG_BIND_READY {
                        self.handle_bind_ready(&task_name, drv_req_write, payload, bundled_fd);
                    } else if header.msg_type == abi::supervisor_protocol::MSG_SERVICE_READY {
                        if let Some(svc) =
                            abi::supervisor_protocol::decode_service_ready_le(payload)
                        {
                            stem::debug!(
                                "SPROUT: SERVICE_READY from {} (ID: {})",
                                task_name,
                                svc.bind_instance_id
                            );
                            info!("SPROUT: Service '{}' is fully operational.", task_name);
                        }
                    } else if header.msg_type == abi::supervisor_protocol::MSG_SERVICE_EXITING {
                        if let Some(svc) =
                            abi::supervisor_protocol::decode_service_exiting_le(payload)
                        {
                            if svc.exit_code == 0 {
                                info!(
                                    "SPROUT: Service '{}' exiting cleanly (ID: {}).",
                                    task_name, svc.bind_instance_id
                                );
                            } else {
                                warn!(
                                    "SPROUT: Service '{}' exiting with error code {} (ID: {}).",
                                    task_name, svc.exit_code, svc.bind_instance_id
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn handle_bind_ready(
        &mut self,
        task_name: &str,
        drv_req_write: ChannelHandle,
        payload: &[u8],
        bundled_fd: u32,
    ) {
        use abi::supervisor_protocol::{self, MSG_BIND_ASSIGNED, MSG_BIND_FAILED, classes};
        use stem::syscall::{channel_send_all, vfs_mount};

        // Helper: send MSG_BIND_FAILED back to the driver.
        let send_failed = |req_write: u32, id: u64, code: u32, msg: &[u8]| {
            let mut reason = [0u8; 64];
            let len = msg.len().min(64);
            reason[..len].copy_from_slice(&msg[..len]);
            let failed = supervisor_protocol::BindFailedPayload {
                bind_instance_id: id,
                error_code: code,
                _reserved: 0,
                reason,
            };
            let mut payload_bytes = [0u8; supervisor_protocol::BIND_FAILED_PAYLOAD_SIZE];
            let mut reply_buf = [0u8; 256];
            if let Some(p_len) =
                supervisor_protocol::encode_bind_failed_le(&failed, &mut payload_bytes)
            {
                if let Some(total_len) = abi::display_driver_protocol::encode_message(
                    &mut reply_buf,
                    MSG_BIND_FAILED,
                    &payload_bytes[..p_len],
                ) {
                    let _ = channel_send_all(req_write, &reply_buf[..total_len]);
                }
            }
        };

        if let Some(ready) = supervisor_protocol::decode_bind_ready_le(payload) {
            let provider_port = bundled_fd;
            if provider_port == 0 {
                warn!("SPROUT: BIND_READY from {} carried no provider FD — rejecting", task_name);
                send_failed(
                    drv_req_write,
                    ready.bind_instance_id,
                    supervisor_protocol::errors::ERR_NO_PROVIDER_HANDLE,
                    b"no provider handle attached",
                );
                return;
            }

            let class_alloc = if ready.class_mask & classes::DISPLAY_CARD != 0 {
                Some(("display", "/dev/display/card"))
            } else if ready.class_mask & classes::INPUT_EVENT != 0 {
                Some(("input", "/dev/input/event"))
            } else if ready.class_mask & classes::BLOCK_DEVICE != 0 {
                Some(("block", "/dev/block/sd"))
            } else if ready.class_mask & classes::NETWORK_INTERFACE != 0 {
                Some(("net", "/dev/net/virtio"))
            } else if ready.class_mask & classes::SOUND_CARD != 0 {
                Some(("sound", "/dev/sound/card"))
            } else {
                None
            };

            let (class_name, root) = match class_alloc {
                Some(pair) => pair,
                None => {
                    warn!(
                        "SPROUT: BIND_READY from {} has unrecognised class_mask 0x{:x} — rejecting",
                        task_name, ready.class_mask
                    );
                    send_failed(
                        drv_req_write,
                        ready.bind_instance_id,
                        supervisor_protocol::errors::ERR_UNKNOWN_CLASS,
                        b"class_mask is zero or unrecognised",
                    );
                    return;
                }
            };

            let mut ledger = self.ledger.lock();
            let unit = ledger.get(class_name).cloned().unwrap_or(0);
            ledger.insert(class_name.to_string(), unit + 1);
            let path = alloc::format!("{}{}", root, unit);
            drop(ledger); // Release ledger lock before mounting

            match vfs_mount(provider_port, &path) {
                Ok(()) => {
                    info!("SPROUT: Sovereign mount success: {} -> {}", task_name, path);

                    let mut assigned = supervisor_protocol::BindAssignedPayload {
                        bind_instance_id: ready.bind_instance_id,
                        status: 0,
                        unit_number: unit,
                        primary_path: [0u8; 64],
                    };
                    let path_bytes = path.as_bytes();
                    let len = path_bytes.len().min(64);
                    assigned.primary_path[..len].copy_from_slice(&path_bytes[..len]);

                    let mut reply_buf = [0u8; 256];
                    let mut payload_bytes = [0u8; supervisor_protocol::BIND_ASSIGNED_PAYLOAD_SIZE];
                    if let Some(p_len) =
                        supervisor_protocol::encode_bind_assigned_le(&assigned, &mut payload_bytes)
                    {
                        if let Some(total_len) = abi::display_driver_protocol::encode_message(
                            &mut reply_buf,
                            MSG_BIND_ASSIGNED,
                            &payload_bytes[..p_len],
                        ) {
                            let _ = channel_send_all(drv_req_write, &reply_buf[..total_len]);
                        }
                    }
                }
                Err(e) => {
                    warn!("SPROUT: Sovereign mount FAILED for {}: {:?}", task_name, e);
                    send_failed(
                        drv_req_write,
                        ready.bind_instance_id,
                        supervisor_protocol::errors::ERR_MOUNT_FAILED,
                        b"vfs_mount failed",
                    );
                }
            }
        } else {
            warn!("SPROUT: Received malformed BIND_READY from {} — rejecting", task_name);
            send_failed(
                drv_req_write,
                0,
                supervisor_protocol::errors::ERR_INVALID_MESSAGE,
                b"BIND_READY payload is malformed",
            );
        }
    }
}
