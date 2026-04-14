//! Handle table: Per-process capability-gated port access
//!
//! Handles are indices into a per-task table that map to ports with
//! specific access modes (read or write).

use super::PortId;

/// A handle is an index into the process handle table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle(pub u32);

/// Access mode for a handle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleMode {
    Read,
    Write,
}

/// Entry in the handle table
#[derive(Debug, Clone, Copy)]
pub struct HandleEntry {
    pub port_id: PortId,
    pub mode: HandleMode,
}

/// Maximum handles per process (v0 limit)
pub const MAX_HANDLES: usize = 1024;

/// Per-process handle table
#[derive(Debug)]
pub struct HandleTable {
    entries: [Option<HandleEntry>; MAX_HANDLES],
}

impl HandleTable {
    /// Create a new empty handle table
    pub const fn new() -> Self {
        Self {
            entries: [None; MAX_HANDLES],
        }
    }

    /// Allocate a new handle for the given port and mode.
    /// Handle 0 is reserved as "invalid" for userspace conventions.
    pub fn alloc(&mut self, port_id: PortId, mode: HandleMode) -> Option<Handle> {
        for (i, slot) in self.entries.iter_mut().enumerate().skip(1) {
            if slot.is_none() {
                *slot = Some(HandleEntry { port_id, mode });
                return Some(Handle(i as u32));
            }
        }
        None // No free slots
    }

    /// Get the entry for a handle, validating mode
    pub fn get(&self, handle: Handle, required_mode: HandleMode) -> Option<&HandleEntry> {
        let idx = handle.0 as usize;
        if idx >= MAX_HANDLES {
            return None;
        }
        self.entries[idx]
            .as_ref()
            .filter(|e| e.mode == required_mode)
    }

    /// Get the entry for a handle without mode validation
    pub fn get_any(&self, handle: Handle) -> Option<&HandleEntry> {
        let idx = handle.0 as usize;
        if idx >= MAX_HANDLES {
            return None;
        }
        self.entries[idx].as_ref()
    }

    /// Close a handle, freeing the slot
    pub fn close(&mut self, handle: Handle) -> Option<HandleEntry> {
        let idx = handle.0 as usize;
        if idx >= MAX_HANDLES {
            return None;
        }
        self.entries[idx].take()
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}
