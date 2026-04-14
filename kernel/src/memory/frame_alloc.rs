use crate::BootModuleDesc;
use crate::PhysRange;
use crate::PhysRangeKind;
use spin::Mutex;

pub static FRAME_ALLOCATOR: FrameAllocatorLocked = FrameAllocatorLocked::new();

pub struct FrameAllocatorLocked(Mutex<Option<FrameAllocator>>);

impl FrameAllocatorLocked {
    pub const fn new() -> Self {
        Self(Mutex::new(None))
    }

    pub unsafe fn init(&self, alloc: FrameAllocator) {
        *self.0.lock() = Some(alloc);
    }

    pub fn with_lock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut FrameAllocator) -> R,
    {
        // For now, we assume single-core boot and no IRQ during early memory init
        let mut lock = self.0.lock();
        f(lock.as_mut().expect("Frame allocator not initialized"))
    }
}

pub struct FrameAllocator {
    bitmap: &'static mut [u64],
    total_frames: usize,
    free_frames: usize,
}

impl FrameAllocator {
    pub fn new_from_boot(
        map: &'static [PhysRange],
        _modules: &'static [BootModuleDesc],
        bitmap: &'static mut [u64],
        hhdm: u64,
    ) -> Self {
        let mut this = Self {
            bitmap,
            total_frames: 0,
            free_frames: 0,
        };

        this.total_frames = this.bitmap.len() * 64;
        this.bitmap.fill(!0);

        for range in map {
            if range.kind == PhysRangeKind::Usable {
                let start = core::cmp::max(range.start, 0);
                if start < range.end {
                    this.mark_free_range(start, range.end);
                }
            }
        }

        let bitmap_phys = (this.bitmap.as_ptr() as u64) - hhdm;
        let bitmap_len = (this.bitmap.len() * 8) as u64;
        this.mark_used_range(bitmap_phys, bitmap_phys + bitmap_len);

        this
    }

    pub fn free_count(&self) -> usize {
        self.free_frames
    }

    pub fn alloc(&mut self) -> Option<(u64,)> {
        for i in 0..self.bitmap.len() {
            if self.bitmap[i] != !0 {
                let bit = self.bitmap[i].trailing_ones() as usize;
                self.bitmap[i] |= 1 << bit;
                self.free_frames -= 1;
                return Some(((i * 64 + bit) as u64 * 4096,));
            }
        }
        None
    }

    /// Allocate `count` physically contiguous frames.
    /// Returns the physical base address of the first frame.
    pub fn alloc_contiguous(&mut self, count: usize) -> Option<u64> {
        if count == 0 {
            return None;
        }
        if count == 1 {
            return self.alloc().map(|f| f.0);
        }

        let total_bits = self.bitmap.len() * 64;
        if count > total_bits {
            return None;
        }

        // Scan for a contiguous run of `count` free bits
        let mut run_start = 0;
        let mut run_len = 0;

        for frame in 0..total_bits {
            let word_idx = frame / 64;
            let bit_idx = frame % 64;
            let is_free = (self.bitmap[word_idx] & (1 << bit_idx)) == 0;

            if is_free {
                if run_len == 0 {
                    run_start = frame;
                }
                run_len += 1;
                if run_len == count {
                    // Found! Mark all as used
                    for f in run_start..(run_start + count) {
                        let wi = f / 64;
                        let bi = f % 64;
                        self.bitmap[wi] |= 1 << bi;
                    }
                    self.free_frames -= count;
                    return Some(run_start as u64 * 4096);
                }
            } else {
                run_len = 0;
            }
        }
        None
    }

    pub fn mark_free_range(&mut self, start: u64, end: u64) {
        let start_page = (start + 4095) / 4096;
        let end_page = end / 4096;
        for p in start_page..end_page {
            let i = (p / 64) as usize;
            let bit = (p % 64) as usize;
            if i < self.bitmap.len() {
                if (self.bitmap[i] & (1 << bit)) != 0 {
                    self.bitmap[i] &= !(1 << bit);
                    self.free_frames += 1;
                }
            }
        }
    }

    pub fn mark_used_range(&mut self, start: u64, end: u64) {
        let start_page = start / 4096;
        let end_page = (end + 4095) / 4096;
        for p in start_page..end_page {
            let i = (p / 64) as usize;
            let bit = (p % 64) as usize;
            if i < self.bitmap.len() {
                if (self.bitmap[i] & (1 << bit)) == 0 {
                    self.bitmap[i] |= 1 << bit;
                    self.free_frames -= 1;
                }
            }
        }
    }
}
