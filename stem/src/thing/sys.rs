// Legacy graph discovery and mutation APIs are eradicated.
// All core system lookups must use VFS paths.

pub use crate::syscall::vfs::{
    vfs_close as close, vfs_open as open, vfs_read as read, vfs_seek as seek, vfs_stat as stat,
    vfs_write as write,
};

pub use crate::syscall::{memfd_create, memfd_phys, vm_map};
