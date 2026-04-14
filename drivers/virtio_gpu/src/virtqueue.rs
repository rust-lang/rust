//! Virtqueue implementation for virtio devices
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use core::ptr::{read_volatile, write_volatile};

/// Virtqueue descriptor entry
#[repr(C, packed)]
pub struct VirtqDesc {
    pub addr: u64,
    pub len: u32,
    pub flags: u16,
    pub next: u16,
}

/// Virtqueue available ring header
#[repr(C, packed)]
pub struct VirtqAvail {
    pub flags: u16,
    pub idx: u16,
    // ring[queue_size] follows
}

/// Virtqueue used ring entry
#[repr(C, packed)]
pub struct VirtqUsedElem {
    pub id: u32,
    pub len: u32,
}

/// Virtqueue used ring header
#[repr(C, packed)]
pub struct VirtqUsed {
    pub flags: u16,
    pub idx: u16,
    // ring[queue_size] follows
}

/// A virtqueue for device communication
pub struct Virtqueue {
    virt_base: u64,
    #[allow(dead_code)]
    phys_base: u64,
    size: u16,
    free_head: u16,
    num_free: u16,
    last_used_idx: u16,
}

impl Virtqueue {
    pub fn new(virt_base: u64, phys_base: u64, size: u16) -> Self {
        // Initialize descriptor table
        let desc_ptr = virt_base as *mut VirtqDesc;
        for i in 0..size {
            unsafe {
                let desc = desc_ptr.add(i as usize);
                write_volatile(&raw mut (*desc).addr, 0);
                write_volatile(&raw mut (*desc).len, 0);
                write_volatile(&raw mut (*desc).flags, 0);
                write_volatile(&raw mut (*desc).next, (i + 1) % size);
            }
        }

        // Initialize avail ring
        let avail_offset = (size as usize) * core::mem::size_of::<VirtqDesc>();
        let avail_ptr = (virt_base + avail_offset as u64) as *mut VirtqAvail;
        unsafe {
            write_volatile(&raw mut (*avail_ptr).flags, 0);
            write_volatile(&raw mut (*avail_ptr).idx, 0);
        }

        // Initialize used ring (after avail ring)
        let used_offset = avail_offset + 6 + (size as usize) * 2;
        let used_ptr = (virt_base + used_offset as u64) as *mut VirtqUsed;
        unsafe {
            write_volatile(&raw mut (*used_ptr).flags, 0);
            write_volatile(&raw mut (*used_ptr).idx, 0);
        }

        Self {
            virt_base,
            phys_base,
            size,
            free_head: 0,
            num_free: size,
            last_used_idx: 0,
        }
    }

    /// Add a buffer chain to the virtqueue
    /// Returns descriptor index or None if queue is full
    pub fn add_buffer(&mut self, bufs: &[(u64, u32, bool)]) -> Option<u16> {
        if bufs.is_empty() || self.num_free < bufs.len() as u16 {
            return None;
        }

        let desc_ptr = self.virt_base as *mut VirtqDesc;
        let head = self.free_head;
        let mut idx = head;

        for (i, (addr, len, writable)) in bufs.iter().enumerate() {
            unsafe {
                let desc = desc_ptr.add(idx as usize);
                write_volatile(&raw mut (*desc).addr, *addr);
                write_volatile(&raw mut (*desc).len, *len);

                let mut flags: u16 = 0;
                if *writable {
                    flags |= crate::virtio::VIRTQ_DESC_F_WRITE;
                }
                if i < bufs.len() - 1 {
                    flags |= crate::virtio::VIRTQ_DESC_F_NEXT;
                }
                write_volatile(&raw mut (*desc).flags, flags);

                idx = read_volatile(&raw const (*desc).next);
            }
        }

        self.free_head = idx;
        self.num_free -= bufs.len() as u16;

        // Add to avail ring
        let avail_offset = (self.size as usize) * core::mem::size_of::<VirtqDesc>();
        let avail_ptr = (self.virt_base + avail_offset as u64) as *mut VirtqAvail;
        unsafe {
            let avail_idx = read_volatile(&raw const (*avail_ptr).idx);
            let ring_ptr = (avail_ptr as *mut u8).add(4) as *mut u16;
            write_volatile(ring_ptr.add((avail_idx % self.size) as usize), head);

            // Memory barrier
            core::sync::atomic::fence(core::sync::atomic::Ordering::Release);

            write_volatile(&raw mut (*avail_ptr).idx, avail_idx.wrapping_add(1));
        }

        Some(head)
    }

    /// Check for completed buffers
    pub fn poll_used(&mut self) -> Option<(u16, u32)> {
        let used_offset =
            (self.size as usize) * core::mem::size_of::<VirtqDesc>() + 6 + (self.size as usize) * 2;
        let used_ptr = (self.virt_base + used_offset as u64) as *mut VirtqUsed;

        unsafe {
            let used_idx = read_volatile(&raw const (*used_ptr).idx);
            if self.last_used_idx == used_idx {
                return None;
            }

            let ring_ptr = (used_ptr as *mut u8).add(4) as *mut VirtqUsedElem;
            let elem = ring_ptr.add((self.last_used_idx % self.size) as usize);
            let id = read_volatile(&raw const (*elem).id);
            let len = read_volatile(&raw const (*elem).len);

            self.last_used_idx = self.last_used_idx.wrapping_add(1);

            // Count chain length by following NEXT flags
            let desc_ptr = self.virt_base as *mut VirtqDesc;
            let mut count = 1u16;
            let mut cur = id as u16;
            loop {
                let flags = read_volatile(&raw const (*desc_ptr.add(cur as usize)).flags);
                if (flags & crate::virtio::VIRTQ_DESC_F_NEXT) == 0 {
                    break;
                }
                let next = read_volatile(&raw const (*desc_ptr.add(cur as usize)).next);
                cur = next;
                count += 1;
            }

            // Return descriptors to free list by linking the last descriptor
            // in the chain to the current free_head
            write_volatile(&raw mut (*desc_ptr.add(cur as usize)).next, self.free_head);
            self.free_head = id as u16;
            self.num_free += count;

            Some((id as u16, len))
        }
    }
}
