use core::{arch::asm, ptr};

use crate::{
    alloc::{GlobalAlloc, Layout, System},
    ptr::null_mut,
    sync::atomic::{AtomicUsize, Ordering},
};


const PAGE_SIZE : usize = 4096;
struct AllocMeta{
    flags : u64,
    size  : usize // this is size of the data total, including alignment but not metadata
}
impl AllocMeta{
    const ALLOCATED_MASK : u64 = 1;
    pub fn new(size : usize, flags : u64) -> Self{
        AllocMeta { flags, size }
    }

    pub fn check_allocated(&self) -> bool{
        self.check_flag(Self::ALLOCATED_MASK)
    }
    pub fn set_allocated(&mut self){
        self.set_flag(Self::ALLOCATED_MASK);
    }


    
    fn check_flag(&self, mask : u64) -> bool{
        (self.flags & mask) > 0
    }
    fn set_flag(&mut self, mask : u64){
        self.flags &= !mask;
        self.flags |= mask;
    }
    fn next(current : *mut Self) -> *mut Self{
        unsafe {
            current.byte_add(current.read().size + size_of::<Self>())
        }
    }
    pub fn split(current : *mut Self, size : usize){
        let mut c = unsafe {
            current.read()
        };
        let next = AllocMeta::next(current);
        let split = AllocMeta::new(c.size-size, c.flags);
        c.set_allocated();
        c.size = size;
        unsafe {
            current.write(c);
            next.write(split);
        }

    }
}

pub struct HobkeyAlloc{
    next_page : AtomicUsize,
    remaining : AtomicUsize
}
impl HobkeyAlloc{
    const HEAP_START : usize = 0x1000;
}


#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HOBKEY_ALLOCATOR.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HOBKEY_ALLOCATOR.dealloc(ptr, layout);
    }
}

pub static HOBKEY_ALLOCATOR : HobkeyAlloc = HobkeyAlloc{
    next_page: AtomicUsize::new(HobkeyAlloc::HEAP_START),
    remaining: AtomicUsize::new(0),
};


fn mmap_syscall(phy : u64, virt : u64, flags : u64) -> i8{
    let ret : i8;
    unsafe {
        asm!(
            "mov rdi,{}",
            "mov rsi, {}",
            "mov rdx, {}",
            "mov eax, 0",
            "int 0x80",
            in(reg) phy,
            in(reg) virt,
            in(reg) flags,
            out("al") ret
        );
    }
    ret
    
}


fn pmm_pop_syscall() -> u64{
    let ret : u64;
    unsafe {
        asm!(
            "mov eax, 1",
            "int 0x80",
            out("rax") ret
        );
    }
    ret
}


 fn pmm_push_syscall(page : u64) -> (){
    unsafe {
        asm!(
            "mov rdi, {}",
            "mov eax, 2",
            "int 0x80",
            in(reg) page
        )
    }
}
#[allow(dead_code)]
mod paging_flags{
    pub const PAGING_PRESENT  : u64  = 1 << 0;  // Present; must be 1 to reference a paging table
    pub const PAGING_RW       : u64  = 1 << 1;  // Read/write; if 0, writes may not be allowed (see Section 4.6)
    pub const PAGING_USER     : u64  = 1 << 2;  // User/supervisor; if 0, user-mode accesses are not allowed (see Section 4.6)
    pub const PAGING_PWT      : u64  = 1 << 3;  // Page-level write-through; indirectly determines memory type (see Section 4.9.2)
    pub const PAGING_PCD      : u64  = 1 << 4;  // Page-level cache disable; indirectly determines memory type (see Section 4.9.2)
    pub const PAGING_ACCESSED : u64  = 1 << 5;  // Accessed; indicates whether this entry has been used (see Section 4.8)
    pub const PAGING_R        : u64  = 1 << 11; // For ordinary paging, ignored; for HLAT paging, restart (see Section 4.8)
}



unsafe impl GlobalAlloc for HobkeyAlloc{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0{
            return null_mut()
        }


        if self.remaining.load(Ordering::Relaxed) < layout.size() + size_of::<AllocMeta>()*2{
            //try to allocate and map another page, if cant return null
            let page = pmm_pop_syscall();
            if page != 0{
                let virt = self.next_page.load(Ordering::Relaxed);
                               
                if mmap_syscall(page, virt as u64, paging_flags::PAGING_RW | paging_flags::PAGING_PRESENT) == -1{
                    pmm_push_syscall(page);
                    return null_mut();
                }
                let _meta : *mut AllocMeta = ptr::with_exposed_provenance_mut(virt);
                unsafe {
                    _meta.write(AllocMeta{
                        flags: 0, size: PAGE_SIZE as usize
                    });
                }
                
                self.next_page.fetch_add(PAGE_SIZE as usize, Ordering::Relaxed);
                self.remaining.fetch_add(PAGE_SIZE as usize, Ordering::Relaxed);
            }
            else{
                return null_mut();
            }
        }

        // split the metadata nodes accordingly
        let current : *mut AllocMeta = ptr::with_exposed_provenance_mut(HobkeyAlloc::HEAP_START);
        loop {
            if !unsafe { current.read().check_allocated() }{
                
                let mut address = (current as usize) + size_of::<AllocMeta>();
                let offset = address % layout.align();
                address += offset;
                AllocMeta::split(current, layout.size()+offset);
                return ptr::with_exposed_provenance_mut(address);
            }
        }     
        
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {
        todo!()
    }
    
    unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
        let size = layout.size();
        // SAFETY: the safety contract for `alloc` must be upheld by the caller.
        let ptr = unsafe { self.alloc(layout) };
        if !ptr.is_null() {
            // SAFETY: as allocation succeeded, the region from `ptr`
            // of size `size` is guaranteed to be valid for writes.
            unsafe { core::ptr::write_bytes(ptr, 0, size) };
        }
        ptr
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: core::alloc::Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller must ensure that the `new_size` does not overflow.
        // `layout.align()` comes from a `Layout` and is thus guaranteed to be valid.
        let new_layout = unsafe { core::alloc::Layout::from_size_align_unchecked(new_size, layout.align()) };
        // SAFETY: the caller must ensure that `new_layout` is greater than zero.
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            // SAFETY: the previously allocated block cannot overlap the newly allocated block.
            // The safety contract for `dealloc` must be upheld by the caller.
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, new_ptr, core::cmp::min(layout.size(), new_size));
                self.dealloc(ptr, layout);
            }
        }
        new_ptr
    }
}