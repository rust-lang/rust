#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use alloc::string::String;

#[derive(Debug, PartialEq, Default)]
pub enum TaskKind {
    #[default]
    App,
    Driver(String),  // Device Kind
    Service(String), // Service Kind
}

#[derive(Default)]
pub struct ManagedTask {
    pub name: String,
    pub kind: TaskKind,
    #[allow(dead_code)]
    pub module_path: String,
    pub pid: Option<u64>,
    pub restarts: u32,
    /// Original argument passed to spawn_process, preserved for restarts
    pub spawn_arg: usize,
    /// Unique token for sovereign registration handshake
    pub bind_instance_id: u64,
    /// Write end of the request channel for handshake response (0 if unused)
    pub drv_req_write: stem::syscall::ChannelHandle,
    /// Read end of the response channel for driver communication (0 if unused)
    pub drv_resp_read: stem::syscall::ChannelHandle,
    /// Bootstrap handle: Read end of req channel (for driver consumption)
    pub boot_req_read: stem::syscall::ChannelHandle,
    /// Bootstrap handle: Write end of resp channel (for driver consumption)
    pub boot_resp_write: stem::syscall::ChannelHandle,
}

impl ManagedTask {
    pub fn new(name: String, kind: TaskKind) -> Self {
        Self {
            name,
            kind,
            ..Default::default()
        }
    }
}
