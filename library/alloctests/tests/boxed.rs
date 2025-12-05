use core::alloc::{AllocError, Allocator, Layout};
use core::cell::Cell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

#[test]
#[expect(dangling_pointers_from_temporaries)]
fn uninitialized_zero_size_box() {
    assert_eq!(
        &*Box::<()>::new_uninit() as *const _,
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[()]>::new_uninit_slice(4).as_ptr(),
        NonNull::<MaybeUninit<()>>::dangling().as_ptr(),
    );
    assert_eq!(
        Box::<[String]>::new_uninit_slice(0).as_ptr(),
        NonNull::<MaybeUninit<String>>::dangling().as_ptr(),
    );
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Dummy {
    _data: u8,
}

#[test]
fn box_clone_and_clone_from_equivalence() {
    for size in (0..8).map(|i| 2usize.pow(i)) {
        let control = vec![Dummy { _data: 42 }; size].into_boxed_slice();
        let clone = control.clone();
        let mut copy = vec![Dummy { _data: 84 }; size].into_boxed_slice();
        copy.clone_from(&control);
        assert_eq!(control, clone);
        assert_eq!(control, copy);
    }
}

/// This test might give a false positive in case the box reallocates,
/// but the allocator keeps the original pointer.
///
/// On the other hand, it won't give a false negative: If it fails, then the
/// memory was definitely not reused.
#[test]
fn box_clone_from_ptr_stability() {
    for size in (0..8).map(|i| 2usize.pow(i)) {
        let control = vec![Dummy { _data: 42 }; size].into_boxed_slice();
        let mut copy = vec![Dummy { _data: 84 }; size].into_boxed_slice();
        let copy_raw = copy.as_ptr() as usize;
        copy.clone_from(&control);
        assert_eq!(copy.as_ptr() as usize, copy_raw);
    }
}

#[test]
fn box_deref_lval() {
    let x = Box::new(Cell::new(5));
    x.set(1000);
    assert_eq!(x.get(), 1000);
}

/// Test that a panic from a destructor does not leak the allocation.
#[test]
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_no_leak() {
    use std::alloc::{AllocError, Allocator, Global, Layout};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr::NonNull;

    struct AllocCount(Cell<i32>);
    unsafe impl Allocator for AllocCount {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0.set(self.0.get() + 1);
            Global.allocate(layout)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            self.0.set(self.0.get() - 1);
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    struct PanicOnDrop {
        _data: u8,
    }
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("PanicOnDrop");
        }
    }

    let alloc = AllocCount(Cell::new(0));
    let b = Box::new_in(PanicOnDrop { _data: 42 }, &alloc);
    assert_eq!(alloc.0.get(), 1);

    let panic_message = catch_unwind(AssertUnwindSafe(|| drop(b))).unwrap_err();
    assert_eq!(*panic_message.downcast_ref::<&'static str>().unwrap(), "PanicOnDrop");
    assert_eq!(alloc.0.get(), 0);
}

#[allow(unused)]
pub struct ConstAllocator;

unsafe impl Allocator for ConstAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(NonNull::slice_from_raw_parts(layout.dangling(), 0)),
            _ => unsafe {
                let ptr = core::intrinsics::const_allocate(layout.size(), layout.align());
                Ok(NonNull::new_unchecked(ptr as *mut [u8; 0] as *mut [u8]))
            },
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, layout: Layout) {
        match layout.size() {
            0 => { /* do nothing */ }
            _ => { /* do nothing too */ }
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        if layout.size() > 0 {
            unsafe {
                ptr.as_mut_ptr().write_bytes(0, layout.size());
            }
        }
        Ok(ptr)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;
        if new_layout.size() > 0 {
            // Safety: `new_ptr` is valid for writes and `ptr` for reads of
            // `old_layout.size()`, because `new_layout.size() >=
            // old_layout.size()` (which is an invariant that must be upheld by
            // callers).
            unsafe {
                new_ptr.as_mut_ptr().copy_from_nonoverlapping(ptr.as_ptr(), old_layout.size());
            }
            // Safety: `ptr` is never used again is also an invariant which must
            // be upheld by callers.
            unsafe {
                self.deallocate(ptr, old_layout);
            }
        }
        Ok(new_ptr)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // Safety: Invariants of `grow_zeroed` and `grow` are the same, and must
        // be enforced by callers.
        let new_ptr = unsafe { self.grow(ptr, old_layout, new_layout)? };
        if new_layout.size() > 0 {
            let old_size = old_layout.size();
            let new_size = new_layout.size();
            let raw_ptr = new_ptr.as_mut_ptr();
            // Safety:
            // - `grow` returned Ok, so the returned pointer must be valid for
            //   `new_size` bytes
            // - `new_size` must be larger than `old_size`, which is an
            //   invariant which must be upheld by callers.
            unsafe {
                raw_ptr.add(old_size).write_bytes(0, new_size - old_size);
            }
        }
        Ok(new_ptr)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;
        if new_layout.size() > 0 {
            // Safety: `new_ptr` and `ptr` are valid for reads/writes of
            // `new_layout.size()` because of the invariants of shrink, which
            // include `new_layout.size()` being smaller than (or equal to)
            // `old_layout.size()`.
            unsafe {
                new_ptr.as_mut_ptr().copy_from_nonoverlapping(ptr.as_ptr(), new_layout.size());
            }
            // Safety: `ptr` is never used again is also an invariant which must
            // be upheld by callers.
            unsafe {
                self.deallocate(ptr, old_layout);
            }
        }
        Ok(new_ptr)
    }

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

#[allow(unused)]
mod pin_coerce_unsized {
    use alloc::boxed::Box;
    use core::pin::Pin;

    trait MyTrait {
        fn action(&self) -> &str;
    }
    impl MyTrait for String {
        fn action(&self) -> &str {
            &*self
        }
    }
    struct MyStruct;
    impl MyTrait for MyStruct {
        fn action(&self) -> &str {
            "MyStruct"
        }
    }

    // Pin coercion should work for Box
    fn pin_box<T: MyTrait + 'static>(arg: Pin<Box<T>>) -> Pin<Box<dyn MyTrait>> {
        arg
    }

    #[test]
    fn pin_coerce_unsized_box() {
        let my_string = "my string";
        let a_string = Box::pin(String::from(my_string));
        let pin_box_str = pin_box(a_string);
        assert_eq!(pin_box_str.as_ref().action(), my_string);
        let a_struct = Box::pin(MyStruct);
        let pin_box_struct = pin_box(a_struct);
        assert_eq!(pin_box_struct.as_ref().action(), "MyStruct");
    }
}
