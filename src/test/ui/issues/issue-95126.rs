// run-pass
// no-prefer-dynamic

// regression test for #95126: io::cleanup() can panic in unusual circumstances

#[global_allocator]
static ALLOCATOR: allocator::TracingSystemAllocator = allocator::TracingSystemAllocator;

fn main() {
    let _ = std::io::stdout();
    allocator::enable_tracing();
    // panic occurs after `main()`
}

// a global allocator that prints on `alloc` and `dealloc`
mod allocator {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::{RefCell, RefMut},
        panic::catch_unwind,
    };

    pub struct TracingSystemAllocator;

    unsafe impl GlobalAlloc for TracingSystemAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);

            let _ = catch_unwind(|| {
                maybe_with_guard(|trace_allocations| {
                    if *trace_allocations {
                        println!("alloc({:?}) = {}", layout, ptr as usize);
                    }
                })
            });

            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);

            let _ = catch_unwind(|| {
                maybe_with_guard(|trace_allocations| {
                    if *trace_allocations {
                        println!("dealloc({}, {:?})", ptr as usize, layout);
                    }
                })
            });
        }
    }

    pub fn enable_tracing() {
        maybe_with_guard(|mut trace| *trace = true)
    }

    // maybe run `f`, if a unique, mutable reference to `TRACE_ALLOCATOR` can be
    // acquired.
    fn maybe_with_guard<F>(f: F)
    where
        F: for<'a> FnOnce(RefMut<'a, bool>),
    {
        let _ = TRACE_ALLOCATOR.try_with(|guard| guard.try_borrow_mut().map(f));
    }

    // used to prevent infinitely recursive tracing
    thread_local! { static TRACE_ALLOCATOR: RefCell<bool> = RefCell::default(); }
}
