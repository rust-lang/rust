//! VirtIO queue (virtqueue) implementation
//!
//! Implements the split virtqueue format used by VirtIO 1.0 devices.
//! Each virtqueue consists of three parts:
//! - Descriptor table
//! - Available ring
//! - Used ring

use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU16, Ordering, fence};

/// Descriptor flags
pub mod desc_flags {
    /// This descriptor continues via the next field
    pub const NEXT: u16 = 1;
    /// This is a write-only descriptor (from driver's perspective)
    pub const WRITE: u16 = 2;
    /// This buffer contains a list of buffer descriptors
    pub const INDIRECT: u16 = 4;
}

/// A single descriptor in the descriptor table
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtqueueDescriptor {
    /// Guest physical address
    pub addr: u64,
    /// Length in bytes
    pub len: u32,
    /// Flags
    pub flags: u16,
    /// Index of next descriptor if NEXT flag is set
    pub next: u16,
}

/// Available ring structure (driver -> device)
#[repr(C)]
struct VirtqueueAvailable {
    flags: u16,
    idx: AtomicU16,
    // Followed by: ring[queue_size], used_event (optional)
}

/// Used ring element
#[repr(C)]
#[derive(Clone, Copy)]
struct VirtqueueUsedElem {
    /// Index of start of used descriptor chain
    id: u32,
    /// Total length written to descriptor chain
    len: u32,
}

/// Used ring structure (device -> driver)
#[repr(C)]
struct VirtqueueUsed {
    flags: u16,
    idx: AtomicU16,
    // Followed by: ring[queue_size], avail_event (optional)
}

/// A virtqueue (split format)
pub struct Virtqueue {
    /// Queue size (must be power of 2)
    queue_size: u16,
    /// Physical address of descriptor table
    desc_phys: u64,
    /// Virtual address of descriptor table
    desc_virt: u64,
    /// Physical address of available ring
    avail_phys: u64,
    /// Virtual address of available ring
    avail_virt: u64,
    /// Physical address of used ring
    used_phys: u64,
    /// Virtual address of used ring
    used_virt: u64,
    /// Next descriptor to allocate
    next_desc: u16,
    /// Last seen used index
    last_used_idx: u16,
    /// Free descriptor list head
    free_head: u16,
    /// Number of free descriptors
    num_free: u16,
}

impl Virtqueue {
    /// Calculate memory requirements for a virtqueue
    pub fn memory_size(queue_size: u16) -> usize {
        let desc_size = core::mem::size_of::<VirtqueueDescriptor>() * queue_size as usize;
        let avail_size = 6 + (2 * queue_size as usize); // flags + idx + ring + used_event
        let used_size = 6 + (8 * queue_size as usize); // flags + idx + ring + avail_event

        // Align each section
        let desc_aligned = (desc_size + 15) & !15;
        let avail_aligned = (avail_size + 1) & !1;
        let used_aligned = (used_size + 3) & !3;

        desc_aligned + avail_aligned + used_aligned
    }

    /// Create a new virtqueue from pre-allocated physically contiguous memory
    ///
    /// # Arguments
    /// * `queue_size` - Number of descriptors (must be power of 2)
    /// * `phys_addr` - Physical base address of queue memory
    /// * `virt_addr` - Virtual address mapping of queue memory
    pub fn new(queue_size: u16, phys_addr: u64, virt_addr: u64) -> Self {
        assert!(
            queue_size.is_power_of_two(),
            "Queue size must be power of 2"
        );
        assert!(queue_size <= 32768, "Queue size too large");

        let desc_size = core::mem::size_of::<VirtqueueDescriptor>() * queue_size as usize;
        let avail_size = 6 + (2 * queue_size as usize);

        let desc_aligned = (desc_size + 15) & !15;
        let avail_aligned = (avail_size + 1) & !1;

        let desc_phys = phys_addr;
        let desc_virt = virt_addr;

        let avail_phys = phys_addr + desc_aligned as u64;
        let avail_virt = virt_addr + desc_aligned as u64;

        let used_phys = avail_phys + avail_aligned as u64;
        let used_virt = avail_virt + avail_aligned as u64;

        // Zero out the memory
        unsafe {
            core::ptr::write_bytes(virt_addr as *mut u8, 0, Self::memory_size(queue_size));
        }

        // Initialize free descriptor list
        let descriptors = unsafe {
            core::slice::from_raw_parts_mut(
                desc_virt as *mut VirtqueueDescriptor,
                queue_size as usize,
            )
        };
        for i in 0..queue_size - 1 {
            descriptors[i as usize].next = i + 1;
        }
        descriptors[(queue_size - 1) as usize].next = 0xFFFF; // End of list

        Self {
            queue_size,
            desc_phys,
            desc_virt,
            avail_phys,
            avail_virt,
            used_phys,
            used_virt,
            next_desc: 0,
            last_used_idx: 0,
            free_head: 0,
            num_free: queue_size,
        }
    }

    /// Get physical addresses for device configuration
    pub fn addresses(&self) -> (u64, u64, u64) {
        (self.desc_phys, self.avail_phys, self.used_phys)
    }

    /// Allocate a descriptor from the free list
    fn alloc_desc(&mut self) -> Option<u16> {
        if self.num_free == 0 {
            return None;
        }

        let desc_idx = self.free_head;
        let descriptors = unsafe {
            core::slice::from_raw_parts_mut(
                self.desc_virt as *mut VirtqueueDescriptor,
                self.queue_size as usize,
            )
        };

        self.free_head = descriptors[desc_idx as usize].next;
        self.num_free -= 1;

        Some(desc_idx)
    }

    /// Free a descriptor back to the free list
    fn free_desc(&mut self, desc_idx: u16) {
        let descriptors = unsafe {
            core::slice::from_raw_parts_mut(
                self.desc_virt as *mut VirtqueueDescriptor,
                self.queue_size as usize,
            )
        };

        descriptors[desc_idx as usize].next = self.free_head;
        self.free_head = desc_idx;
        self.num_free += 1;
    }

    /// Add a buffer to the virtqueue
    ///
    /// Returns the descriptor index (for tracking)
    pub fn add_buffer(&mut self, phys_addr: u64, len: u32, write: bool) -> Option<u16> {
        let desc_idx = self.alloc_desc()?;

        let descriptors = unsafe {
            core::slice::from_raw_parts_mut(
                self.desc_virt as *mut VirtqueueDescriptor,
                self.queue_size as usize,
            )
        };

        descriptors[desc_idx as usize] = VirtqueueDescriptor {
            addr: phys_addr,
            len,
            flags: if write { desc_flags::WRITE } else { 0 },
            next: 0,
        };

        Some(desc_idx)
    }

    /// Make buffers available to the device
    pub fn kick(&mut self, desc_idx: u16) {
        let avail = unsafe { &*(self.avail_virt as *const VirtqueueAvailable) };
        let ring_ptr = (self.avail_virt + 4) as *mut u16;

        // Get current index
        let idx = avail.idx.load(Ordering::Acquire);

        // Add descriptor to available ring
        unsafe {
            write_volatile(
                ring_ptr.add((idx as usize) % self.queue_size as usize),
                desc_idx,
            );
        }

        // Update index
        fence(Ordering::Release);
        avail.idx.store(idx.wrapping_add(1), Ordering::Release);
    }

    /// Check if there are used buffers available
    pub fn has_used(&self) -> bool {
        let used = unsafe { &*(self.used_virt as *const VirtqueueUsed) };
        let used_idx = used.idx.load(Ordering::Acquire);
        used_idx != self.last_used_idx
    }

    /// Get next used buffer
    ///
    /// Returns (descriptor_idx, bytes_written)
    pub fn get_used(&mut self) -> Option<(u16, u32)> {
        if !self.has_used() {
            return None;
        }

        let used = unsafe { &*(self.used_virt as *const VirtqueueUsed) };
        let ring_ptr = (self.used_virt + 4) as *const VirtqueueUsedElem;

        let idx = self.last_used_idx as usize % self.queue_size as usize;
        let elem = unsafe { read_volatile(ring_ptr.add(idx)) };

        self.last_used_idx = self.last_used_idx.wrapping_add(1);

        Some((elem.id as u16, elem.len))
    }

    /// Reclaim a descriptor (put it back in the free list)
    pub fn reclaim(&mut self, desc_idx: u16) {
        self.free_desc(desc_idx);
    }

    /// Get number of free descriptors
    pub fn num_free(&self) -> u16 {
        self.num_free
    }
}
