use std::alloc::Layout;
use std::ptr::NonNull;

use nix::sys::mman;
use rustc_index::bit_set::DenseBitSet;

/// How many bytes of memory each bit in the bitset represents.
const COMPRESSION_FACTOR: usize = 4;

/// A dedicated allocator for interpreter memory contents, ensuring they are stored on dedicated
/// pages (not mixed with Miri's own memory). This is used in native-lib mode.
#[derive(Debug)]
pub struct IsolatedAlloc {
    /// Pointers to page-aligned memory that has been claimed by the allocator.
    /// Every pointer here must point to a page-sized allocation claimed via
    /// mmap. These pointers are used for "small" allocations.
    page_ptrs: Vec<NonNull<u8>>,
    /// Metadata about which bytes have been allocated on each page. The length
    /// of this vector must be the same as that of `page_ptrs`, and the domain
    /// size of the bitset must be exactly `page_size / COMPRESSION_FACTOR`.
    ///
    /// Conceptually, each bit of the bitset represents the allocation status of
    /// one n-byte chunk on the corresponding element of `page_ptrs`. Thus,
    /// indexing into it should be done with a value one-nth of the corresponding
    /// offset on the matching `page_ptrs` element (n = `COMPRESSION_FACTOR`).
    page_infos: Vec<DenseBitSet<usize>>,
    /// Pointers to multiple-page-sized allocations. These must also be page-aligned,
    /// with their size stored as the second element of the vector.
    huge_ptrs: Vec<(NonNull<u8>, usize)>,
    /// The host (not emulated) page size.
    page_size: usize,
}

impl IsolatedAlloc {
    /// Creates an empty allocator.
    pub fn new() -> Self {
        Self {
            page_ptrs: Vec::new(),
            huge_ptrs: Vec::new(),
            page_infos: Vec::new(),
            // SAFETY: `sysconf(_SC_PAGESIZE)` is always safe to call at runtime
            // See https://www.man7.org/linux/man-pages/man3/sysconf.3.html
            page_size: unsafe { libc::sysconf(libc::_SC_PAGESIZE).try_into().unwrap() },
        }
    }

    /// For simplicity, we serve small allocations in multiples of COMPRESSION_FACTOR
    /// bytes with at least that alignment.
    #[inline]
    fn normalized_layout(layout: Layout) -> Layout {
        let align =
            if layout.align() < COMPRESSION_FACTOR { COMPRESSION_FACTOR } else { layout.align() };
        let size = layout.size().next_multiple_of(COMPRESSION_FACTOR);
        Layout::from_size_align(size, align).unwrap()
    }

    /// For greater-than-page-sized allocations, returns the allocation size we need to request
    /// including the slack we need to satisfy the alignment request.
    #[inline]
    fn huge_normalized_layout(&self, layout: Layout) -> usize {
        // Allocate in page-sized chunks
        let size = layout.size().next_multiple_of(self.page_size);
        // And make sure the align is at least one page
        let align = std::cmp::max(layout.align(), self.page_size);
        // pg_count gives us the # of pages needed to satisfy the size. For
        // align > page_size where align = n * page_size, a sufficently-aligned
        // address must exist somewhere in the range of
        // some_page_aligned_address..some_page_aligned_address + (n-1) * page_size
        // (since if some_page_aligned_address + n * page_size is sufficently aligned,
        // then so is some_page_aligned_address itself per the definition of n, so we
        // can avoid using that 1 extra page).
        // Thus we allocate n-1 extra pages
        let pg_count = size.div_ceil(self.page_size);
        let extra_pages = align.strict_div(self.page_size).saturating_sub(1);

        pg_count.strict_add(extra_pages).strict_mul(self.page_size)
    }

    /// Determined whether a given normalized (size, align) should be sent to
    /// `alloc_huge` / `dealloc_huge`.
    #[inline]
    fn is_huge_alloc(&self, layout: &Layout) -> bool {
        layout.align() > self.page_size / 2 || layout.size() >= self.page_size / 2
    }

    /// Allocates memory as described in `Layout`. This memory should be deallocated
    /// by calling `dealloc` on this same allocator.
    ///
    /// SAFETY: See `alloc::alloc()`.
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by caller
        unsafe { self.allocate(layout, false) }
    }

    /// Same as `alloc`, but zeroes out the memory.
    ///
    /// SAFETY: See `alloc::alloc_zeroed()`.
    pub unsafe fn alloc_zeroed(&mut self, layout: Layout) -> *mut u8 {
        // SAFETY: Upheld by caller
        unsafe { self.allocate(layout, true) }
    }

    /// Abstracts over the logic of `alloc_zeroed` vs `alloc`, as determined by
    /// the `zeroed` argument.
    ///
    /// SAFETY: See `alloc::alloc()`.
    unsafe fn allocate(&mut self, layout: Layout, zeroed: bool) -> *mut u8 {
        let layout = IsolatedAlloc::normalized_layout(layout);
        if self.is_huge_alloc(&layout) {
            // SAFETY: Validity of `layout` upheld by caller; we checked that
            // the size and alignment are appropriate for being a huge alloc
            unsafe { self.alloc_huge(layout) }
        } else {
            for (&mut page, pinfo) in std::iter::zip(&mut self.page_ptrs, &mut self.page_infos) {
                // SAFETY: The value in `self.page_size` is used to allocate
                // `page`, with page alignment
                if let Some(ptr) =
                    unsafe { Self::alloc_small(self.page_size, layout, page, pinfo, zeroed) }
                {
                    return ptr;
                }
            }

            // We get here only if there's no space in our existing pages
            let page_size = self.page_size;
            // Add another page and allocate from it; this cannot fail since the
            // new page is empty and we already asserted it fits into a page
            let (page, pinfo) = self.add_page();

            // SAFETY: See comment on `alloc_from_page` above
            unsafe { Self::alloc_small(page_size, layout, page, pinfo, zeroed).unwrap() }
        }
    }

    /// Used internally by `allocate` to abstract over some logic.
    ///
    /// SAFETY: `page` must be a page-aligned pointer to an allocated page,
    /// where the allocation is (at least) `page_size` bytes.
    unsafe fn alloc_small(
        page_size: usize,
        layout: Layout,
        page: NonNull<u8>,
        pinfo: &mut DenseBitSet<usize>,
        zeroed: bool,
    ) -> Option<*mut u8> {
        // Check every alignment-sized block and see if there exists a `size`
        // chunk of empty space i.e. forall idx . !pinfo.contains(idx / n)
        for offset in (0..page_size).step_by(layout.align()) {
            let offset_pinfo = offset / COMPRESSION_FACTOR;
            let size_pinfo = layout.size() / COMPRESSION_FACTOR;
            // DenseBitSet::contains() panics if the index is out of bounds
            if pinfo.domain_size() < offset_pinfo + size_pinfo {
                break;
            }
            if !pinfo.contains_any(offset_pinfo..offset_pinfo + size_pinfo) {
                pinfo.insert_range(offset_pinfo..offset_pinfo + size_pinfo);
                // SAFETY: We checked the available bytes after `idx` in the call
                // to `domain_size` above and asserted there are at least `idx +
                // layout.size()` bytes available and unallocated after it.
                // `page` must point to the start of the page, so adding `idx`
                // is safe per the above.
                unsafe {
                    let ptr = page.add(offset);
                    if zeroed {
                        // Only write the bytes we were specifically asked to
                        // zero out, even if we allocated more
                        ptr.write_bytes(0, layout.size());
                    }
                    return Some(ptr.as_ptr());
                }
            }
        }
        None
    }

    /// Expands the available memory pool by adding one page.
    fn add_page(&mut self) -> (NonNull<u8>, &mut DenseBitSet<usize>) {
        // SAFETY: mmap is always safe to call when requesting anonymous memory
        let page_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                self.page_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
            .cast::<u8>()
        };
        assert_ne!(page_ptr.addr(), usize::MAX, "mmap failed");
        // `page_infos` has to have one bit for each `COMPRESSION_FACTOR`-sized chunk of bytes in the page.
        assert!(self.page_size.is_multiple_of(COMPRESSION_FACTOR));
        self.page_infos.push(DenseBitSet::new_empty(self.page_size / COMPRESSION_FACTOR));
        self.page_ptrs.push(NonNull::new(page_ptr).unwrap());
        (NonNull::new(page_ptr).unwrap(), self.page_infos.last_mut().unwrap())
    }

    /// Allocates in multiples of one page on the host system.
    /// Will always be zeroed.
    ///
    /// SAFETY: Same as `alloc()`.
    unsafe fn alloc_huge(&mut self, layout: Layout) -> *mut u8 {
        let size = self.huge_normalized_layout(layout);
        // SAFETY: mmap is always safe to call when requesting anonymous memory
        let ret = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
            .cast::<u8>()
        };
        assert_ne!(ret.addr(), usize::MAX, "mmap failed");
        self.huge_ptrs.push((NonNull::new(ret).unwrap(), size));
        // huge_normalized_layout ensures that we've overallocated enough space
        // for this to be valid.
        ret.map_addr(|a| a.next_multiple_of(layout.align()))
    }

    /// Deallocates a pointer from this allocator.
    ///
    /// SAFETY: This pointer must have been allocated by calling `alloc()` (or
    /// `alloc_zeroed()`) with the same layout as the one passed on this same
    /// `IsolatedAlloc`.
    pub unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        let layout = IsolatedAlloc::normalized_layout(layout);

        if self.is_huge_alloc(&layout) {
            // SAFETY: Partly upheld by caller, and we checked that the size
            // and align, meaning this must have been allocated via `alloc_huge`
            unsafe {
                self.dealloc_huge(ptr, layout);
            }
        } else {
            // SAFETY: It's not a huge allocation, therefore it is a small one.
            let idx = unsafe { self.dealloc_small(ptr, layout) };

            // This may have been the last allocation on this page. If so, free the entire page.
            // FIXME: this can lead to threshold effects, we should probably add some form
            // of hysteresis.
            if self.page_infos[idx].is_empty() {
                self.page_infos.remove(idx);
                let page_ptr = self.page_ptrs.remove(idx);
                // SAFETY: We checked that there are no outstanding allocations
                // from us pointing to this page, and we know it was allocated
                // in add_page as exactly a single page.
                unsafe {
                    assert_eq!(libc::munmap(page_ptr.as_ptr().cast(), self.page_size), 0);
                }
            }
        }
    }

    /// Returns the index of the page that this was deallocated from.
    ///
    /// SAFETY: the pointer must have been allocated with `alloc_small`.
    unsafe fn dealloc_small(&mut self, ptr: *mut u8, layout: Layout) -> usize {
        // Offset of the pointer in the current page
        let offset = ptr.addr() % self.page_size;
        // And then the page's base address
        let page_addr = ptr.addr() - offset;

        // Find the page this allocation belongs to.
        // This could be made faster if the list was sorted -- the allocator isn't fully optimized at the moment.
        let pinfo = std::iter::zip(&mut self.page_ptrs, &mut self.page_infos)
            .enumerate()
            .find(|(_, (page, _))| page.addr().get() == page_addr);
        let Some((idx_of_pinfo, (_, pinfo))) = pinfo else {
            panic!("Freeing in an unallocated page: {ptr:?}\nHolding pages {:?}", self.page_ptrs)
        };
        // Mark this range as available in the page.
        let ptr_idx_pinfo = offset / COMPRESSION_FACTOR;
        let size_pinfo = layout.size() / COMPRESSION_FACTOR;
        for idx in ptr_idx_pinfo..ptr_idx_pinfo + size_pinfo {
            pinfo.remove(idx);
        }
        idx_of_pinfo
    }

    /// SAFETY: Same as `dealloc()` with the added requirement that `layout`
    /// must ask for a size larger than the host pagesize.
    unsafe fn dealloc_huge(&mut self, ptr: *mut u8, layout: Layout) {
        let size = self.huge_normalized_layout(layout);
        // Find the huge allocation containing `ptr`.
        let idx = self
            .huge_ptrs
            .iter()
            .position(|&(pg, size)| {
                pg.addr().get() <= ptr.addr() && ptr.addr() < pg.addr().get().strict_add(size)
            })
            .expect("Freeing unallocated pages");
        // And kick it from the list
        let (un_offset_ptr, size2) = self.huge_ptrs.remove(idx);
        assert_eq!(size, size2, "got wrong layout in dealloc");
        // SAFETY: huge_ptrs contains allocations made with mmap with the size recorded there.
        unsafe {
            let ret = libc::munmap(un_offset_ptr.as_ptr().cast(), size);
            assert_eq!(ret, 0);
        }
    }

    /// Returns a vector of page addresses managed by the allocator.
    pub fn pages(&self) -> Vec<usize> {
        let mut pages: Vec<usize> =
            self.page_ptrs.iter().map(|p| p.expose_provenance().get()).collect();
        for (ptr, size) in self.huge_ptrs.iter() {
            for i in 0..size / self.page_size {
                pages.push(ptr.expose_provenance().get().strict_add(i * self.page_size));
            }
        }
        pages
    }

    /// Protects all owned memory as `PROT_NONE`, preventing accesses.
    ///
    /// SAFETY: Accessing memory after this point will result in a segfault
    /// unless it is first unprotected.
    pub unsafe fn prepare_ffi(&mut self) -> Result<(), nix::errno::Errno> {
        let prot = mman::ProtFlags::PROT_NONE;
        unsafe { self.mprotect(prot) }
    }

    /// Deprotects all owned memory by setting it to RW. Erroring here is very
    /// likely unrecoverable, so it may panic if applying those permissions
    /// fails.
    pub fn unprep_ffi(&mut self) {
        let prot = mman::ProtFlags::PROT_READ | mman::ProtFlags::PROT_WRITE;
        unsafe {
            self.mprotect(prot).unwrap();
        }
    }

    /// Applies `prot` to every page managed by the allocator.
    ///
    /// SAFETY: Accessing memory in violation of the protection flags will
    /// trigger a segfault.
    unsafe fn mprotect(&mut self, prot: mman::ProtFlags) -> Result<(), nix::errno::Errno> {
        for &pg in &self.page_ptrs {
            unsafe {
                mman::mprotect(pg.cast(), self.page_size, prot)?;
            }
        }
        for &(hpg, size) in &self.huge_ptrs {
            unsafe {
                mman::mprotect(hpg.cast(), size.next_multiple_of(self.page_size), prot)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to assert that all bytes from `ptr` to `ptr.add(layout.size())`
    /// are zeroes.
    ///
    /// SAFETY: `ptr` must have been allocated with `layout`.
    unsafe fn assert_zeroes(ptr: *mut u8, layout: Layout) {
        // SAFETY: Caller ensures this is valid
        unsafe {
            for ofs in 0..layout.size() {
                assert_eq!(0, ptr.add(ofs).read());
            }
        }
    }

    /// Check that small (sub-pagesize) allocations are properly zeroed out.
    #[test]
    fn small_zeroes() {
        let mut alloc = IsolatedAlloc::new();
        // 256 should be less than the pagesize on *any* system
        let layout = Layout::from_size_align(256, 32).unwrap();
        // SAFETY: layout size is the constant above, not 0
        let ptr = unsafe { alloc.alloc_zeroed(layout) };
        // SAFETY: `ptr` was just allocated with `layout`
        unsafe {
            assert_zeroes(ptr, layout);
            alloc.dealloc(ptr, layout);
        }
    }

    /// Check that huge (> 1 page) allocations are properly zeroed out also.
    #[test]
    fn huge_zeroes() {
        let mut alloc = IsolatedAlloc::new();
        // 16k is about as big as pages get e.g. on macos aarch64
        let layout = Layout::from_size_align(16 * 1024, 128).unwrap();
        // SAFETY: layout size is the constant above, not 0
        let ptr = unsafe { alloc.alloc_zeroed(layout) };
        // SAFETY: `ptr` was just allocated with `layout`
        unsafe {
            assert_zeroes(ptr, layout);
            alloc.dealloc(ptr, layout);
        }
    }

    /// Check that repeatedly reallocating the same memory will still zero out
    /// everything properly
    #[test]
    fn repeated_allocs() {
        let mut alloc = IsolatedAlloc::new();
        // Try both sub-pagesize allocs and those larger than / equal to a page
        for sz in (1..=(16 * 1024)).step_by(128) {
            let layout = Layout::from_size_align(sz, 1).unwrap();
            // SAFETY: all sizes in the range above are nonzero as we start from 1
            let ptr = unsafe { alloc.alloc_zeroed(layout) };
            // SAFETY: `ptr` was just allocated with `layout`, which was used
            // to bound the access size
            unsafe {
                assert_zeroes(ptr, layout);
                ptr.write_bytes(255, sz);
                alloc.dealloc(ptr, layout);
            }
        }
    }

    /// Checks that allocations of different sizes do not overlap, then for memory
    /// leaks that might have occurred.
    #[test]
    fn check_leaks_and_overlaps() {
        let mut alloc = IsolatedAlloc::new();

        // Some random sizes and aligns
        let mut sizes = vec![32; 10];
        sizes.append(&mut vec![15; 4]);
        sizes.append(&mut vec![256; 12]);
        // Give it some multi-page ones too
        sizes.append(&mut vec![32 * 1024; 4]);
        sizes.push(4 * 1024);

        // Matching aligns for the sizes
        let mut aligns = vec![16; 12];
        aligns.append(&mut vec![256; 2]);
        aligns.append(&mut vec![64; 12]);
        aligns.append(&mut vec![4096; 4]);
        // And one that requests align > page_size
        aligns.push(64 * 1024);

        // Make sure we didn't mess up in the test itself!
        assert_eq!(sizes.len(), aligns.len());

        // Aggregate the sizes and aligns into a vec of layouts, then allocate them
        let layouts: Vec<_> = std::iter::zip(sizes, aligns)
            .map(|(sz, al)| Layout::from_size_align(sz, al).unwrap())
            .collect();
        // SAFETY: all sizes specified in `sizes` are nonzero
        let ptrs: Vec<_> =
            layouts.iter().map(|layout| unsafe { alloc.alloc_zeroed(*layout) }).collect();

        for (&ptr, &layout) in std::iter::zip(&ptrs, &layouts) {
            // We requested zeroed allocations, so check that that's true
            // Then write to the end of the current size, so if the allocs
            // overlap (or the zeroing is wrong) then `assert_zeroes` will panic.
            // Also check that the alignment we asked for was respected
            assert_eq!(ptr.addr().strict_rem(layout.align()), 0);
            // SAFETY: each `ptr` was allocated with its corresponding `layout`,
            // which is used to bound the access size
            unsafe {
                assert_zeroes(ptr, layout);
                ptr.write_bytes(255, layout.size());
                alloc.dealloc(ptr, layout);
            }
        }

        // And then verify that no memory was leaked after all that
        assert!(alloc.page_ptrs.is_empty() && alloc.huge_ptrs.is_empty());
    }
}
