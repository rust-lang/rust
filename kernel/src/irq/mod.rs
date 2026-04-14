//! Kernel IRQ Subsystem
//!
//! Manages interrupt routing and userspace IRQ subscriptions.

use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;

pub mod msi;

pub static IRQ_DISABLE_HOOK: core::sync::atomic::AtomicPtr<()> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());
pub static IRQ_RESTORE_HOOK: core::sync::atomic::AtomicPtr<()> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

pub unsafe fn irq_disable_erased() -> crate::IrqState {
    let ptr = IRQ_DISABLE_HOOK.load(Ordering::SeqCst);
    if !ptr.is_null() {
        let hook: fn() -> crate::IrqState = core::mem::transmute(ptr);
        hook()
    } else {
        crate::IrqState(0)
    }
}

pub unsafe fn irq_restore_erased(state: crate::IrqState) {
    let ptr = IRQ_RESTORE_HOOK.load(Ordering::SeqCst);
    if !ptr.is_null() {
        let hook: fn(crate::IrqState) = core::mem::transmute(ptr);
        hook(state);
    }
}

pub const EXTERNAL_VECTOR_START: u8 = 0x40;
pub const EXTERNAL_VECTOR_END: u8 = 0xEF;

/// Maximum supported vectors for IRQ dispatch
pub const MAX_VECTORS: usize = 256;
pub const MAX_SUBSCRIBERS_PER_VECTOR: usize = 4;

/// IRQ slot for a single vector
struct IrqSlot {
    /// Task IDs subscribed to this vector (0 means empty)
    subscribers: [AtomicU64; MAX_SUBSCRIBERS_PER_VECTOR],
    /// Pending interrupt counts per subscriber
    pending_counts: [AtomicU32; MAX_SUBSCRIBERS_PER_VECTOR],
}

impl IrqSlot {
    const fn new() -> Self {
        Self {
            subscribers: [const { AtomicU64::new(0) }; MAX_SUBSCRIBERS_PER_VECTOR],
            pending_counts: [const { AtomicU32::new(0) }; MAX_SUBSCRIBERS_PER_VECTOR],
        }
    }
}

/// Global IRQ registry (lock-free for IRQ context)
pub struct IrqRegistry {
    slots: [IrqSlot; MAX_VECTORS],
}

impl IrqRegistry {
    pub const fn new() -> Self {
        Self {
            slots: [const { IrqSlot::new() }; MAX_VECTORS],
        }
    }

    /// Subscribe a task to receive interrupts for a vector
    pub fn subscribe(&self, vector: u8, task_id: u64) -> Result<(), ()> {
        let _lock = REGISTRY_MUTEX.lock();
        let slot = &self.slots[vector as usize];

        let mut first_empty = None;
        for i in 0..MAX_SUBSCRIBERS_PER_VECTOR {
            let sub = slot.subscribers[i].load(Ordering::Relaxed);
            if sub == task_id {
                return Err(()); // Already subscribed
            }
            if sub == 0 && first_empty.is_none() {
                first_empty = Some(i);
            }
        }

        if let Some(idx) = first_empty {
            slot.pending_counts[idx].store(0, Ordering::Relaxed);
            slot.subscribers[idx].store(task_id, Ordering::Release);
            Ok(())
        } else {
            Err(()) // Out of slots
        }
    }

    /// Unsubscribe a task from a vector
    pub fn unsubscribe(&self, vector: u8, task_id: u64) {
        let _lock = REGISTRY_MUTEX.lock();
        let slot = &self.slots[vector as usize];
        for i in 0..MAX_SUBSCRIBERS_PER_VECTOR {
            if slot.subscribers[i].load(Ordering::Relaxed) == task_id {
                slot.subscribers[i].store(0, Ordering::Relaxed);
                slot.pending_counts[i].store(0, Ordering::Relaxed);
            }
        }
    }

    /// Dispatch an interrupt - increment pending count and wake waiters
    /// This MUST be IRQ-safe (lock-free)
    pub fn dispatch(&self, vector: u8) {
        let slot = &self.slots[vector as usize];
        for i in 0..MAX_SUBSCRIBERS_PER_VECTOR {
            let task_id = slot.subscribers[i].load(Ordering::Acquire);
            if task_id != 0 {
                slot.pending_counts[i].fetch_add(1, Ordering::Relaxed);
                // Wake the task using type-erased hook
                unsafe {
                    crate::sched::wake_task_erased(task_id);
                }
            }
        }
    }

    /// Wait for interrupt - returns pending count and resets it
    pub fn try_wait(&self, vector: u8, task_id: u64) -> u32 {
        let slot = &self.slots[vector as usize];
        for i in 0..MAX_SUBSCRIBERS_PER_VECTOR {
            if slot.subscribers[i].load(Ordering::Acquire) == task_id {
                return slot.pending_counts[i].swap(0, Ordering::SeqCst);
            }
        }
        0
    }

    pub fn poll_pending(&self, vector: u8, task_id: u64) -> Option<u32> {
        let slot = &self.slots[vector as usize];
        for i in 0..MAX_SUBSCRIBERS_PER_VECTOR {
            if slot.subscribers[i].load(Ordering::Acquire) == task_id {
                return Some(slot.pending_counts[i].load(Ordering::Acquire));
            }
        }
        None
    }
}

/// Mutex for protecting registry mutations (subscription/unsubscription)
/// This is only used in task context.
static REGISTRY_MUTEX: Mutex<()> = Mutex::new(());

/// Global IRQ registry instance
pub static IRQ_REGISTRY: IrqRegistry = IrqRegistry::new();

pub struct VectorAllocator {
    used: [bool; MAX_VECTORS],
    owner_device: [u64; MAX_VECTORS],
    owner_irq: [u8; MAX_VECTORS],
}

impl VectorAllocator {
    pub const fn new() -> Self {
        Self {
            used: [false; MAX_VECTORS],
            owner_device: [0; MAX_VECTORS],
            owner_irq: [0; MAX_VECTORS],
        }
    }

    pub fn alloc(&mut self, resource_id: u64, irq_index: u8) -> Option<u8> {
        for v in EXTERNAL_VECTOR_START..=EXTERNAL_VECTOR_END {
            let idx = v as usize;
            if !self.used[idx] {
                self.used[idx] = true;
                self.owner_device[idx] = resource_id;
                self.owner_irq[idx] = irq_index;
                return Some(v);
            }
        }
        None
    }

    pub fn free(&mut self, vector: u8) {
        let idx = vector as usize;
        if idx < MAX_VECTORS {
            self.used[idx] = false;
            self.owner_device[idx] = 0;
            self.owner_irq[idx] = 0;
        }
    }

    pub fn owner(&self, vector: u8) -> Option<(u64, u8)> {
        let idx = vector as usize;
        if idx < MAX_VECTORS && self.used[idx] && self.owner_device[idx] != 0 {
            return Some((self.owner_device[idx], self.owner_irq[idx]));
        }
        None
    }
}

pub static VECTOR_ALLOC: Mutex<VectorAllocator> = Mutex::new(VectorAllocator::new());

/// Called from interrupt handlers to dispatch IRQ
pub fn dispatch_irq(vector: u8) {
    crate::trace::irq_ring::push(abi::trace::TraceEvent::Irq {
        vector,
        timestamp: crate::trace::now(),
    });
    IRQ_REGISTRY.dispatch(vector);
}

pub fn alloc_vector(resource_id: u64, irq_index: u8) -> Option<u8> {
    VECTOR_ALLOC.lock().alloc(resource_id, irq_index)
}

pub fn free_vector(vector: u8) {
    VECTOR_ALLOC.lock().free(vector);
}

pub fn vector_owner(vector: u8) -> Option<(u64, u8)> {
    VECTOR_ALLOC.lock().owner(vector)
}

/// Subscribe current task to a vector
pub fn subscribe(vector: u8) -> Result<(), ()> {
    let task_id = unsafe { crate::sched::current_tid_current() };
    IRQ_REGISTRY.subscribe(vector, task_id)
}

/// Wait for IRQ - blocks until interrupt fires
/// Returns number of pending interrupts
pub fn wait(vector: u8) -> u32 {
    let task_id = unsafe { crate::sched::current_tid_current() };

    loop {
        let count = IRQ_REGISTRY.try_wait(vector, task_id);
        if count > 0 {
            return count;
        }

        // Block until woken by interrupt
        unsafe {
            crate::sched::block_current_erased();
        }
    }
}

pub fn poll(vector: u8) -> Option<u32> {
    let task_id = unsafe { crate::sched::current_tid_current() };
    IRQ_REGISTRY.poll_pending(vector, task_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU32, Ordering};

    static WAKE_COUNT: AtomicU32 = AtomicU32::new(0);

    fn mock_wake(id: u64) {
        let _ = id;
        WAKE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn test_irq_lock_free_dispatch() {
        let registry = IrqRegistry::new();
        let vector = 0x42;

        WAKE_COUNT.store(0, Ordering::Relaxed);
        unsafe {
            crate::sched::blocking::WAKE_TASK_HOOK.store(mock_wake as *mut (), Ordering::SeqCst);
        }

        registry.subscribe(vector, 1).expect("Subscribe task 1");
        registry.subscribe(vector, 2).expect("Subscribe task 2");

        registry.dispatch(vector);

        assert_eq!(WAKE_COUNT.load(Ordering::Relaxed), 2);
        assert_eq!(registry.try_wait(vector, 1), 1);
        assert_eq!(registry.try_wait(vector, 2), 1);

        WAKE_COUNT.store(0, Ordering::Relaxed);

        registry.dispatch(vector);
        registry.dispatch(vector);
        assert_eq!(registry.try_wait(vector, 1), 2);
    }

    #[test]
    fn test_irq_deadlock_prevention() {
        let registry = IrqRegistry::new();
        let vector = 0x43;
        registry.subscribe(vector, 10).unwrap();

        // Lock the subscription mutex
        let _lock = REGISTRY_MUTEX.lock();

        // Dispatch should still work (lock-free)
        registry.dispatch(vector);

        assert_eq!(registry.try_wait(vector, 10), 1);
    }
}
