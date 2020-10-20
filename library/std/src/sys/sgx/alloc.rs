use super::ext::arch;
use crate::alloc::{GlobalAlloc, Layout, System};
use crate::cmp;
use crate::fmt::{self, Debug, Formatter};
use crate::marker::PhantomData;
use crate::mem;
use crate::ptr;
use crate::sys::sgx::abi::{mem as sgx_mem, usercalls};
use core::sync::atomic::{AtomicBool, Ordering};

use super::waitqueue::SpinMutex;
use sgx_isa::{PageType, Secinfo, SecinfoFlags};

// Using a SpinMutex because we never want to exit the enclave waiting for the
// allocator.
//
// The current allocator here is the `dlmalloc` crate which we've got included
// in the rust-lang/rust repository as a submodule. The crate is a port of
// dlmalloc.c from C to Rust.
#[cfg_attr(test, linkage = "available_externally")]
#[export_name = "_ZN16__rust_internals3std3sys3sgx5alloc8DLMALLOCE"]
static DLMALLOC: SpinMutex<dlmalloc::Dlmalloc<Sgx>> =
    SpinMutex::new(dlmalloc::Dlmalloc::new_with_allocator(Sgx {}));

/// System interface implementation for SGX platform
struct Sgx;

impl Sgx {
    const PAGE_SIZE: usize = 0x1000;

    unsafe fn allocator() -> &'static mut SGXv2Allocator {
        static mut SGX2_ALLOCATOR: SGXv2Allocator = SGXv2Allocator::new();
        unsafe { &mut SGX2_ALLOCATOR }
    }
}

unsafe impl dlmalloc::Allocator for Sgx {
    /// Allocs system resources
    fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
        static INIT: AtomicBool = AtomicBool::new(false);
        if size <= sgx_mem::heap_size() {
            // No ordering requirement since this function is protected by the global lock.
            if !INIT.swap(true, Ordering::Relaxed) {
                return (sgx_mem::heap_base() as _, sgx_mem::heap_size(), 0);
            }
        }

        match unsafe { Sgx::allocator().alloc(size) } {
            Some(base) => (base, size, 0),
            None => (ptr::null_mut(), 0, 0),
        }
    }

    fn remap(&self, _ptr: *mut u8, _oldsize: usize, _newsize: usize, _can_move: bool) -> *mut u8 {
        ptr::null_mut()
    }

    fn free_part(&self, ptr: *mut u8, oldsize: usize, newsize: usize) -> bool {
        assert_eq!(oldsize % Sgx::PAGE_SIZE, 0);
        assert_eq!(newsize % Sgx::PAGE_SIZE, 0);
        unsafe { Sgx::allocator().free_part(ptr, oldsize, newsize).is_ok() }
    }

    fn free(&self, ptr: *mut u8, size: usize) -> bool {
        if !sgx_mem::is_unmapped_range(ptr, size) {
            return false;
        }
        assert_eq!(size % Sgx::PAGE_SIZE, 0);
        unsafe { Sgx::allocator().free(ptr, size).is_ok() }
    }

    fn can_release_part(&self, _flags: u32) -> bool {
        true
    }

    fn allocates_zeros(&self) -> bool {
        false
    }

    fn page_size(&self) -> usize {
        Sgx::PAGE_SIZE
    }
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller must uphold the safety contract for `malloc`
        unsafe { DLMALLOC.lock().malloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller must uphold the safety contract for `malloc`
        unsafe { DLMALLOC.lock().calloc(layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller must uphold the safety contract for `malloc`
        unsafe { DLMALLOC.lock().free(ptr, layout.size(), layout.align()) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller must uphold the safety contract for `malloc`
        unsafe { DLMALLOC.lock().realloc(ptr, layout.size(), layout.align(), new_size) }
    }
}

// The following functions are needed by libunwind. These symbols are named
// in pre-link args for the target specification, so keep that in sync.
#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_c_alloc(size: usize, align: usize) -> *mut u8 {
    unsafe { crate::alloc::alloc(Layout::from_size_align_unchecked(size, align)) }
}

#[cfg(not(test))]
#[no_mangle]
pub unsafe extern "C" fn __rust_c_dealloc(ptr: *mut u8, size: usize, align: usize) {
    unsafe { crate::alloc::dealloc(ptr, Layout::from_size_align_unchecked(size, align)) }
}

struct SGXv2Allocator(Option<BuddyAllocator>);
unsafe impl Send for SGXv2Allocator {}

impl SGXv2Allocator {
    pub const fn new() -> SGXv2Allocator {
        SGXv2Allocator(None)
    }

    fn allocator(&mut self) -> &mut BuddyAllocator {
        if self.0.is_none() {
            let region_base = sgx_mem::unmapped_base();
            let region_size = sgx_mem::unmapped_size();
            self.0 =
                Some(BuddyAllocator::new(region_base as _, region_size as _, Sgx::PAGE_SIZE).unwrap());
        }
        self.0.as_mut().unwrap()
    }

    pub unsafe fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        self.allocator().alloc::<Sgx2Mapper>(size).ok()
    }

    pub unsafe fn free(&mut self, ptr: *mut u8, size: usize) -> Result<(), Error> {
        self.allocator().free::<Sgx2Mapper>(ptr, size, 0)
    }

    pub unsafe fn free_part(
        &mut self,
        ptr: *mut u8,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), Error> {
        self.allocator().free::<Sgx2Mapper>(ptr, old_size, new_size)
    }
}

struct Sgx2Mapper;

impl MemoryMapper for Sgx2Mapper {
    fn map_region(base: *const u8, size: usize) -> Result<(), Error> {
        assert_eq!(size % Sgx::PAGE_SIZE, 0);
        let flags = SecinfoFlags::from(PageType::Reg)
            | SecinfoFlags::R
            | SecinfoFlags::W
            | SecinfoFlags::PENDING;
        let secinfo = Secinfo::from(flags).into();
        for offset in (0..size as isize).step_by(Sgx::PAGE_SIZE) {
            let page = unsafe { base.offset(offset) };

            // In order to add a new page, the OS needs to issue an `eaug` instruction, after which the enclave
            // needs to accept the changes with an `eaccept`. The sgx driver at time of writing only issues an `eaug`
            // when a #PF within the enclave occured due to unmapped memory. By issuing an `eaccept` on
            // unmapped memory, we force such a #PF. Eventually the `eaccept` instruction will be
            // re-executed and succeed.
            arch::eaccept(page as _, &secinfo).map_err(|_| Error::MapFailed)?;
        }

        Ok(())
    }

    fn unmap_region(base: *const u8, size: usize) -> Result<(), Error> {
        fn accept_trim(base: *const u8, size: usize) -> Result<(), Error> {
            let flags = SecinfoFlags::from(PageType::Trim) | SecinfoFlags::MODIFIED;
            let secinfo = Secinfo::from(flags).into();

            for offset in (0..size as isize).step_by(Sgx::PAGE_SIZE) {
                let page = unsafe { base.offset(offset) };
                arch::eaccept(page as _, &secinfo).map_err(|_| Error::UnmapFailed)?;
            }
            Ok(())
        }

        assert_eq!(size % Sgx::PAGE_SIZE, 0);
        // Signal to OS that pages are no longer used and should be trimmed
        usercalls::trim(base, size).map_err(|_| Error::UnmapFailed)?;
        // Accept removing of pages
        accept_trim(base, size).map_err(|_| Error::UnmapFailed)?;
        // Let the OS remove the pages
        usercalls::remove_trimmed(base, size).map_err(|_| Error::UnmapFailed)?;
        Ok(())
    }

    fn page_size() -> usize {
        Sgx::PAGE_SIZE
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    AlignmentError,
    FreeGrowsAllocation,
    SizeNotSupported,
    DoubleFree,
    MemoryNotManagedByAllocator,
    MemorySizeNotPowerOfTwo,
    MinBlockSizeLargerThanMemory,
    MinBlockSizeTooSmall,
    MapFailed,
    UnmapFailed,
    OutOfMemory,
}

pub trait MemoryMapper {
    fn map_region(base: *const u8, size: usize) -> Result<(), Error>;

    fn unmap_region(base: *const u8, size: usize) -> Result<(), Error>;

    fn page_size() -> usize;
}

/// A small, simple allocator that can only allocate blocks of a pre-determined, specific size.
#[derive(Debug, PartialEq, Eq)]
pub struct SimpleAllocator<T> {
    memory: Region,
    free_blocks: *mut u8,
    next_uninit_block: *mut u8,
    phantom: PhantomData<T>,
}

impl<T> SimpleAllocator<T> {
    pub fn block_size() -> usize {
        let t_size = mem::size_of::<T>();
        let p_size = mem::size_of::<*mut u8>();
        cmp::max(t_size, p_size).next_power_of_two()
    }

    pub fn new(memory_base: usize, memory_size: usize) -> Result<SimpleAllocator<T>, Error> {
        if memory_base % Self::block_size() != 0 {
            return Err(Error::AlignmentError);
        }
        Ok(SimpleAllocator {
            memory: Region { addr: memory_base as _, size: memory_size },
            next_uninit_block: memory_base as _,
            free_blocks: ptr::null_mut(),
            phantom: PhantomData,
        })
    }

    pub fn alloc<M: MemoryMapper>(&mut self, content: T) -> Result<*mut T, Error> {
        if (self.memory.addr as usize) % M::page_size() != 0
            || M::page_size() % Self::block_size() != 0
        {
            return Err(Error::AlignmentError);
        }

        unsafe {
            if self.free_blocks.is_null() {
                let ptr = self.next_uninit_block as *mut T;
                if (ptr as *const u8) < self.memory.end() {
                    // There are no free memory blocks, but part of the memory region is still
                    // uninitialized; use a new uninitialized block
                    if (ptr as usize) % M::page_size() == 0 {
                        // Request that a new page is mapped in memory
                        M::map_region(ptr as _, M::page_size())?;
                    }
                    self.next_uninit_block =
                        (self.next_uninit_block as usize + Self::block_size()) as *mut u8;
                    assert_eq!((ptr as usize) % Self::block_size(), 0);
                    ptr::write(ptr, content);
                    Ok(ptr)
                } else {
                    Err(Error::OutOfMemory)
                }
            } else if self.next_uninit_block < self.memory.end() {
                // There are free memory blocks available, recycle one
                let new_head: *mut u8 = ptr::read(self.free_blocks as _);
                let ptr: *mut T = self.free_blocks as _;
                self.free_blocks = new_head;
                assert_eq!((ptr as usize) % Self::block_size(), 0);
                ptr::write(ptr, content);
                Ok(ptr)
            } else {
                Err(Error::OutOfMemory)
            }
        }
    }

    pub fn free(&mut self, ptr: *mut T) {
        unsafe {
            ptr::write(ptr as _, self.free_blocks);
            self.free_blocks = ptr as _;
        }
    }
}

#[derive(PartialEq)]
pub enum Block {
    Free,
    Allocated,
    Partitioned(*mut Block, *mut Block),
}

impl Debug for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Block::Allocated => f.pad("A"),
            Block::Free => f.pad("F"),
            Block::Partitioned(l, r) => unsafe {
                let s = format!("({:?}, {:?})", *l, *r);
                f.pad(&s)
            },
        }
    }
}

#[derive(Debug)]
pub struct BuddyAllocator {
    block: *mut Block,
    min_block_size: usize,
    memory: Region,
    allocator: SimpleAllocator<Block>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    addr: *mut u8,
    size: usize,
}

impl Region {
    fn new(addr: *mut u8, size: usize) -> Region {
        Region { addr, size }
    }

    fn split(&self) -> (Region, Region) {
        let left = Region { addr: self.addr, size: self.size / 2 };
        let right =
            Region { addr: (left.addr as usize + left.size) as _, size: self.size - left.size };
        (left, right)
    }

    fn join(&self, other: &Region) -> Option<Region> {
        let start0 = cmp::min(self.addr, other.addr);
        let start1 = cmp::max(self.addr, other.addr);
        let end0 = cmp::min(self.end(), other.end());
        let end1 = cmp::max(self.end(), other.end());

        if end0 == start1 {
            Some(Region { addr: start0, size: end1 as usize - start0 as usize })
        } else {
            None
        }
    }

    fn intersect(&self, other: &Region) -> Option<Region> {
        let start = crate::cmp::max(self.addr, other.addr);
        let end = crate::cmp::min(self.end(), other.end());
        if start < end {
            Some(Region { addr: start, size: end as usize - start as usize })
        } else {
            None
        }
    }

    fn subtract(&self, other: &Region) -> Option<Region> {
        if other.size == 0 {
            return Some(self.to_owned());
        }
        if self.addr < other.addr {
            let start = self.addr;
            let end = crate::cmp::min(self.end() as usize, other.addr as usize);

            if start as usize != end {
                return Some(Region { addr: start, size: end - start as usize });
            }
        } else {
            if other.end() < self.end() {
                return Some(Region {
                    addr: other.end(),
                    size: self.end() as usize - other.end() as usize,
                });
            }
        }
        None
    }

    fn end(&self) -> *mut u8 {
        (self.addr as usize + self.size) as _
    }

    fn contains(&self, ptr: *mut u8) -> bool {
        self.addr <= ptr && ptr < self.end()
    }
}

impl BuddyAllocator {
    fn tree_depth(memory_size: usize, min_block_size: usize) -> u32 {
        let max_depth = memory_size.next_power_of_two().trailing_zeros();
        let block_depth = min_block_size.next_power_of_two().trailing_zeros();

        assert!(min_block_size <= memory_size);
        max_depth - block_depth
    }

    fn max_metadata_entries(memory_size: usize, min_block_size: usize) -> u32 {
        let depth = Self::tree_depth(memory_size, min_block_size);
        (0x1u32 << (depth + 1)) - 1
    }

    fn max_metadata_size(memory_size: usize, min_block_size: usize) -> usize {
        // The algorithm sometimes temporarily uses 1 additional allocation, we need to account for
        // that
        (Self::max_metadata_entries(memory_size, min_block_size) as usize + 1)
            * SimpleAllocator::<Block>::block_size()
    }

    pub fn new(
        memory_base: usize,
        memory_size: usize,
        min_block_size: usize,
    ) -> Result<BuddyAllocator, Error> {
        if !memory_size.is_power_of_two() {
            return Err(Error::MemorySizeNotPowerOfTwo);
        }
        if !min_block_size.is_power_of_two() {
            return Err(Error::MemorySizeNotPowerOfTwo);
        }
        if memory_size < min_block_size {
            return Err(Error::MinBlockSizeLargerThanMemory);
        }
        if memory_size < Self::max_metadata_size(memory_size, min_block_size) {
            return Err(Error::MinBlockSizeTooSmall);
        }

        let allocator = SimpleAllocator::new(
            memory_base,
            Self::max_metadata_size(memory_size, min_block_size).next_power_of_two(),
        )?;
        let buddy = BuddyAllocator {
            block: ptr::null_mut(),
            min_block_size,
            memory: Region::new(memory_base as _, memory_size),
            allocator,
        };
        Ok(buddy)
    }

    unsafe fn alloc_ex<M: MemoryMapper>(
        &mut self,
        memory: Region,
        block: *mut Block,
        alloc_size: usize,
        map_memory: bool,
    ) -> Result<Region, Error> {
        unsafe {
            assert!(self.min_block_size <= memory.size);
            if memory.size < alloc_size {
                return Err(Error::OutOfMemory);
            }

            match ptr::read(block) {
                Block::Free => {
                    if 2 * alloc_size <= memory.size && self.min_block_size * 2 <= memory.size {
                        // Very large free block found, split region recursively
                        let left = self.allocator.alloc::<M>(Block::Free)?;
                        let right = self.allocator.alloc::<M>(Block::Free)?;
                        *block = Block::Partitioned(left, right);
                        self.alloc_ex::<M>(memory, block, alloc_size, map_memory)
                    } else {
                        // Small free block is found. May split it up further to reduce internal fragmentation
                        if (memory.size - alloc_size) < self.min_block_size
                            || memory.size < 2 * self.min_block_size
                        {
                            // Use entire region
                            ptr::write(block, Block::Allocated);
                            if map_memory {
                                // Don't map metadata in memory. The SimpleAllocator will take care of
                                // that
                                M::map_region(memory.addr, memory.size)?;
                            }
                            Ok(memory)
                        } else {
                            // Split block
                            let block_left = self.allocator.alloc::<M>(Block::Free)?;
                            let block_right = self.allocator.alloc::<M>(Block::Free)?;
                            ptr::write(block, Block::Partitioned(block_left, block_right));
                            let (memory_left, memory_right) = memory.split();
                            let left_size = memory_left.size;
                            let alloc_left =
                                self.alloc_ex::<M>(memory_left, block_left, left_size, map_memory)?;
                            let alloc_right = self.alloc_ex::<M>(
                                memory_right,
                                block_right,
                                alloc_size - alloc_left.size,
                                map_memory,
                            )?;
                            // `alloc_left` should have received a complete block. `alloc_right` will
                            // only receive a chunk of the available mememory but as we favor the
                            // beginning of memory both chunks should be adjacent
                            Ok(alloc_left
                                .join(&alloc_right)
                                .expect("Bug: could not join adjacent regions"))
                        }
                    }
                }
                Block::Partitioned(block_left, block_right) => {
                    let (memory_left, memory_right) = memory.split();
                    if let Ok(left) =
                        self.alloc_ex::<M>(memory_left, block_left, alloc_size, map_memory)
                    {
                        Ok(left)
                    } else if let Ok(right) =
                        self.alloc_ex::<M>(memory_right, block_right, alloc_size, map_memory)
                    {
                        Ok(right)
                    } else {
                        Err(Error::OutOfMemory)
                    }
                }
                Block::Allocated => Err(Error::OutOfMemory),
            }
        }
    }

    pub fn alloc<M: MemoryMapper>(&mut self, size: usize) -> Result<*mut u8, Error> {
        if self.min_block_size < M::page_size() {
            return Err(Error::MinBlockSizeTooSmall);
        }
        if self.block.is_null() {
            // Reserve space for own book keeping
            self.block = self.allocator.alloc::<M>(Block::Free)?;
            let metadata = unsafe {
                self.alloc_ex::<M>(
                    self.memory.to_owned(),
                    self.block,
                    Self::max_metadata_size(self.memory.size, self.min_block_size),
                    false,
                )
            };
            assert!(metadata.is_ok());
        }

        let region = unsafe { self.alloc_ex::<M>(self.memory.to_owned(), self.block, size, true)? };
        Ok(region.addr)
    }

    unsafe fn free_ex<M: MemoryMapper>(
        &mut self,
        block: *mut Block,
        memory: &Region,
        free: &Region,
    ) -> Result<(), Error> {
        unsafe {
            match ptr::read(block) {
                Block::Allocated => {
                    if let Some(_alloc) = memory.subtract(free) {
                        // Split block into two allocated regions and continue freeing recursively
                        assert_eq!(_alloc.addr, memory.addr);
                        let left = self.allocator.alloc::<M>(Block::Allocated)?;
                        let right = self.allocator.alloc::<M>(Block::Allocated)?;
                        *block = Block::Partitioned(left, right);
                        self.free_ex::<M>(block, memory, free)
                    } else {
                        // Free entire memory block
                        ptr::write(block, Block::Free);
                        if M::page_size() < memory.size {
                            M::unmap_region(memory.addr, memory.size)?;
                        }
                        Ok(())
                    }
                }
                Block::Partitioned(block_left, block_right) => {
                    let (memory_left, memory_right) = memory.split();
                    if let Some(overlap) = memory_right.intersect(free) {
                        self.free_ex::<M>(block_right, &memory_right, &overlap)?;
                    }
                    if let Some(overlap) = memory_left.intersect(free) {
                        self.free_ex::<M>(block_left, &memory_left, &overlap)?;
                    }
                    if ptr::read(block_left) == Block::Free && ptr::read(block_right) == Block::Free
                    {
                        self.allocator.free(block_left);
                        self.allocator.free(block_right);
                        ptr::write(block, Block::Free);
                        if M::page_size() == memory.size {
                            // The left and right parts combined are exactly one page. At a lower
                            // level, it couldn't be unmapped as there still may be data on that
                            // page. Now the entire page is free, unmap it. It also isn't possible
                            // that the block size is larger than a page as the buddy allocator
                            // always halfs the available memory. If the block now spans two pages,
                            // it would already have been unmapped on a lower level
                            M::unmap_region(memory.addr, memory.size)?;
                        }
                    }
                    Ok(())
                }
                Block::Free => Err(Error::DoubleFree),
            }
        }
    }

    pub fn free<M: MemoryMapper>(
        &mut self,
        ptr: *mut u8,
        old_size: usize,
        new_size: usize,
    ) -> Result<(), Error> {
        if !self.memory.contains(ptr) {
            return Err(Error::MemoryNotManagedByAllocator);
        }
        if old_size < new_size {
            return Err(Error::FreeGrowsAllocation);
        }
        assert_eq!(old_size % M::page_size(), 0);
        if new_size % M::page_size() != 0 {
            return Err(Error::SizeNotSupported);
        };
        if new_size % self.min_block_size != 0 {
            return Err(Error::SizeNotSupported);
        };
        let old_alloc = Region::new(ptr, old_size);
        let new_alloc = Region::new(ptr, new_size);
        let free = old_alloc.subtract(&new_alloc).ok_or(Error::SizeNotSupported)?;
        let memory = self.memory.to_owned();
        unsafe { self.free_ex::<M>(self.block, &memory, &free) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{BuddyAllocator, Error, MemoryMapper, Region, SimpleAllocator};
    use std::alloc::GlobalAlloc;

    pub struct Linux;

    impl MemoryMapper for Linux {
        fn map_region(base: *const u8, size: usize) {
            if base as usize % Self::page_size() != 0 {
                panic!("Cannot map a page at {:x?}", base);
            }
            if size as usize % Self::page_size() != 0 {
                panic!("Cannot map a page of {}", size);
            }
            assert_eq!(size % Self::page_size(), 0);
            unsafe {
                libc::mprotect(base as _, size, libc::PROT_READ | libc::PROT_WRITE);
            }
        }

        fn unmap_region(base: *const u8, size: usize) {
            assert_eq!(size % Self::page_size(), 0);
            unsafe {
                libc::mprotect(base as _, size, libc::PROT_NONE);
            }
        }

        fn page_size() -> usize {
            0x1000
        }
    }

    #[test]
    fn region_subtract() {
        let block0 = Region { addr: 0x10_000 as _, size: 0x1000 };
        let block1 = Region { addr: 0x11_000 as _, size: 0x2000 };
        let block2 = Region { addr: 0x12_000 as _, size: 0x4000 };
        let block3 = Region { addr: 0x13_000 as _, size: 0x2000 };
        let block4 = Region { addr: 0x14_000 as _, size: 0x6000 };
        let block_null0 = Region { addr: 0x10_000 as _, size: 0 };
        let block_null1 = Region { addr: 0x14_800 as _, size: 0 };
        let block_null2 = Region { addr: 0x11_000 as _, size: 0 };
        assert_eq!(block1.subtract(&block0), Some(block1.clone()));
        assert_eq!(block1.subtract(&block3), Some(block1.clone()));
        assert_eq!(block1.subtract(&block1), None);
        assert_eq!(block2.subtract(&block1), Some(Region { addr: 0x13_000 as _, size: 0x3000 }));
        assert_eq!(block2.subtract(&block4), Some(Region { addr: 0x12_000 as _, size: 0x2000 }));
        assert_eq!(block2.subtract(&block3), Some(Region { addr: 0x12_000 as _, size: 0x1000 }));
        assert_eq!(block3.subtract(&block2), None);
        assert_eq!(block4.subtract(&block2), Some(Region { addr: 0x16_000 as _, size: 0x4000 }));
        assert_eq!(block0.subtract(&block_null0), Some(block0.clone()));
        assert_eq!(block0.subtract(&block_null1), Some(block0.clone()));
        assert_eq!(block0.subtract(&block_null2), Some(block0.clone()));
    }

    #[test]
    fn region_join() {
        let block0 = Region { addr: 0x10_000 as _, size: 0x1000 };
        let block1 = Region { addr: 0x11_000 as _, size: 0x2000 };
        let block2 = Region { addr: 0x12_000 as _, size: 0x4000 };
        let block_null0 = Region { addr: 0x10_000 as _, size: 0 };
        let block_null2 = Region { addr: 0x11_000 as _, size: 0 };
        let block01 = Region { addr: 0x10_000 as _, size: 0x3000 };
        assert_eq!(block0.join(&block1), Some(block01.clone()));
        assert_eq!(block1.join(&block0), Some(block01.clone()));
        assert_eq!(block0.join(&block2), None);
        assert_eq!(block2.join(&block0), None);
        assert_eq!(block_null0.join(&block0), Some(block0.clone()));
        assert_eq!(block_null2.join(&block0), Some(block0.clone()));
    }

    #[test]
    fn region_intersect() {
        let block1 = Region { addr: 0x11_000 as _, size: 0x2000 };
        let block2 = Region { addr: 0x12_000 as _, size: 0x4000 };
        let block3 = Region { addr: 0x13_000 as _, size: 0x2000 };
        let block12 = Region { addr: 0x12_000 as _, size: 0x1000 };
        assert_eq!(block1.intersect(&block2), Some(block12.clone()));
        assert_eq!(block2.intersect(&block1), Some(block12.clone()));
        assert_eq!(block3.intersect(&block2), Some(block3.clone()));
        assert_eq!(block2.intersect(&block3), Some(block3.clone()));
        assert_eq!(block3.intersect(&block1), None);
        assert_eq!(block1.intersect(&block3), None);
        assert_eq!(block1.intersect(&block1), Some(block1.clone()));
    }

    #[test]
    fn tree_depth() {
        assert_eq!(BuddyAllocator::tree_depth(1, 1), 0);
        assert_eq!(BuddyAllocator::tree_depth(8, 1), 3);
        assert_eq!(BuddyAllocator::tree_depth(16, 1), 4);
        assert_eq!(BuddyAllocator::tree_depth(16, 2), 3);
        assert_eq!(BuddyAllocator::tree_depth(16, 4), 2);
    }

    #[test]
    fn buddy_alloc() {
        unsafe {
            let memory_size = 0x10000;
            let memory_base = std::alloc::System
                .alloc(std::alloc::Layout::from_size_align(memory_size, memory_size).unwrap());
            Linux::unmap_region(memory_base, memory_size);
            let mut space = BuddyAllocator::new(memory_base as _, memory_size, 0x1000).unwrap();
            let alloc0 = space.alloc::<Linux>(0x511);
            let alloc1 = space.alloc::<Linux>(0x511);
            assert_eq!(Ok(Region::new((memory_base as usize + 0x1000) as _, 0x1000)), alloc0);
            assert_eq!(Ok(Region::new((memory_base as usize + 0x2000) as _, 0x1000)), alloc1);
            assert_eq!(Ok(()), space.free::<Linux>(alloc1.unwrap().addr, 0x1000, 0));
            assert_eq!(Ok(()), space.free::<Linux>(alloc0.unwrap().addr, 0x1000, 0));
        }
    }

    #[test]
    fn buddy_alloc2() {
        unsafe {
            let memory_size = 0x10000;
            let memory_base = std::alloc::System
                .alloc(std::alloc::Layout::from_size_align(memory_size, memory_size).unwrap());
            Linux::unmap_region(memory_base, memory_size);
            let mut space = BuddyAllocator::new(memory_base as _, memory_size, 0x1000).unwrap();
            let r = space.alloc::<Linux>(0x8000).unwrap();
            assert_eq!(format!("{:?}", *space.block), "((((A, F), F), F), A)");
            assert_eq!(Ok(()), space.free::<Linux>(r.addr, 0x8000, 0x4000));
            assert_eq!(format!("{:?}", *space.block), "((((A, F), F), F), (A, F))");
            assert_eq!(Ok(()), space.free::<Linux>(r.addr, 0x4000, 0x1000));
            assert_eq!(format!("{:?}", *space.block), "((((A, F), F), F), (((A, F), F), F))");
            assert_eq!(Ok(()), space.free::<Linux>(r.addr, 0x1000, 0));
            assert_eq!(format!("{:?}", *space.block), "((((A, F), F), F), F)");

            let r0 = space.alloc::<Linux>(0x2000).unwrap();
            assert_eq!(format!("{:?}", *space.block), "((((A, F), A), F), F)");

            let r1 = space.alloc::<Linux>(0x8000).unwrap();
            assert_eq!(format!("{:?}", *space.block), "((((A, F), A), F), A)");

            let mut r2 = space.alloc::<Linux>(0x4000).unwrap();
            assert_eq!(format!("{:?}", *space.block), "((((A, F), A), A), A)");

            let r3 = space.alloc::<Linux>(0x1000).unwrap();
            assert_eq!(format!("{:?}", *space.block), "((((A, A), A), A), A)");
            assert_eq!(space.alloc::<Linux>(0x1000), Err(Error::OutOfMemory));

            let new_size = 0x1000;
            assert_eq!(Ok(()), space.free::<Linux>(r2.addr, r2.size, new_size));
            r2.size = new_size;
            assert_eq!(format!("{:?}", *space.block), "((((A, A), A), ((A, F), F)), A)");

            assert!(space.free::<Linux>(r0.addr, r0.size, 0).is_ok());
            assert_eq!(format!("{:?}", *space.block), "((((A, A), F), ((A, F), F)), A)");

            assert!(space.free::<Linux>(r1.addr, r1.size, 0).is_ok());
            assert_eq!(format!("{:?}", *space.block), "((((A, A), F), ((A, F), F)), F)");

            assert!(space.free::<Linux>(r2.addr, r2.size, 0).is_ok());
            assert_eq!(format!("{:?}", *space.block), "((((A, A), F), F), F)");

            assert!(space.free::<Linux>(r3.addr, r3.size, 0).is_ok());
            assert_eq!(format!("{:?}", *space.block), "((((A, F), F), F), F)");
        }
    }

    #[test]
    pub fn buddy_alloc_bruteforce() {
        fn mark_allocated(base: *mut u8, size: usize) {
            for index in 0..size {
                let ptr = (base as usize + index) as *mut u8;
                unsafe {
                    assert_eq!(*ptr, 0);
                    *ptr = 1;
                }
            }
        }

        fn mark_free(base: *mut u8, size: usize) {
            for index in 0..size {
                let ptr = (base as usize + index) as *mut u8;
                unsafe {
                    assert_eq!(*ptr, 1);
                    *ptr = 0;
                }
            }
        }

        use rand::Rng;

        let memory_size = 1 * 1024 * 1024;
        let memory_base = unsafe {
            std::alloc::System.alloc_zeroed(
                std::alloc::Layout::from_size_align(memory_size, memory_size).unwrap(),
            )
        };
        Linux::unmap_region(memory_base, memory_size);
        let mut space = BuddyAllocator::new(memory_base as _, memory_size, 0x1000).unwrap();
        let mut rnd = rand::thread_rng();
        let mut pointers: Vec<(*mut u8, usize)> = Vec::new();

        for _i in 0..1000 {
            if rnd.gen() {
                // Allocate
                let size = rnd.gen::<usize>() % (memory_size / 10);
                if let Ok(region) = space.alloc::<Linux>(size) {
                    mark_allocated(region.addr, region.size);
                    pointers.push((region.addr, region.size));
                }
            } else {
                // Free
                if 0 < pointers.len() {
                    let idx = rnd.gen::<usize>() % pointers.len();
                    let (ptr, size) = pointers.remove(idx);
                    mark_free(ptr, size);
                    assert_eq!(Ok(()), space.free::<Linux>(ptr, size, 0));
                }
            }
        }

        while let Some((ptr, size)) = pointers.pop() {
            mark_free(ptr, size);
            assert_eq!(Ok(()), space.free::<Linux>(ptr, size, 0));
        }
    }

    #[test]
    fn simple_alloc() {
        unsafe {
            let region = std::alloc::System
                .alloc(std::alloc::Layout::from_size_align(0x1000, 0x1000).unwrap());
            Linux::unmap_region(region, 0x1000);
            let mut allocator = SimpleAllocator::<u32>::new(region as _, 0x1000).unwrap();
            let mut ptrs = Vec::new();
            for i in 0..100 {
                let ptr = allocator.alloc::<Linux>(i).unwrap();
                assert!(
                    (region as *mut u32) <= ptr && ptr < (region as usize + 0x1000) as *mut u32
                );
                ptrs.push(ptr);
            }
            for ptr in ptrs.iter() {
                allocator.free(*ptr);
            }
        }
    }

    #[test]
    fn bruteforce_simple_alloc() {
        fn mark_allocated(base: *mut u8, size: usize) {
            for index in 0..size {
                let ptr = (base as usize + index) as *mut u8;
                unsafe {
                    *ptr = 1;
                }
            }
        }

        fn mark_free(base: *mut u8, size: usize) {
            for index in 0..size {
                let ptr = (base as usize + index) as *mut u8;
                unsafe {
                    assert_eq!(*ptr, 1);
                    *ptr = 0;
                }
            }
        }

        use rand::Rng;
        use std::alloc::GlobalAlloc;

        let memory_size = 20 * 1024 * 1024;
        let region = unsafe {
            std::alloc::System.alloc_zeroed(
                std::alloc::Layout::from_size_align(memory_size, memory_size.next_power_of_two())
                    .unwrap(),
            )
        };

        Linux::unmap_region(region, memory_size);
        let mut space = SimpleAllocator::<u32>::new(region as _, memory_size).unwrap();
        let mut rnd = rand::thread_rng();
        let mut ptrs = Vec::new();
        let num_runs = 10000;
        for i in 0..num_runs {
            let force_free = (9 * num_runs) / 10 < i;
            if rnd.gen::<usize>() % 100 < 70 && !force_free {
                // alloc
                match space.alloc::<Linux>(0) {
                    Ok(ptr) => {
                        ptrs.push(ptr);
                        assert!(ptr < (region as usize + memory_size) as _);
                        assert!(region <= ptr as _);
                        mark_allocated(ptr as _, SimpleAllocator::<u32>::block_size());
                    }
                    Err(Error::OutOfMemory) => (),
                    _ => panic!("Unexpected error"),
                }
            } else {
                // free
                if 0 < ptrs.len() {
                    let idx = rnd.gen::<usize>() % ptrs.len();
                    let ptr = ptrs.remove(idx);
                    mark_free(ptr as _, SimpleAllocator::<u32>::block_size());
                    space.free(ptr);
                }
            }
        }
    }
}
