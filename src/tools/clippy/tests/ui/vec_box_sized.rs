//@no-rustfix

#![allow(dead_code)]
#![feature(allocator_api)]

use std::alloc::{AllocError, Allocator, Layout};
use std::ptr::NonNull;

struct SizedStruct(i32);
struct UnsizedStruct([i32]);
struct BigStruct([i32; 10000]);

struct DummyAllocator;
unsafe impl Allocator for DummyAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        todo!()
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        todo!()
    }
}

/// The following should trigger the lint
mod should_trigger {
    use super::{DummyAllocator, SizedStruct};
    const C: Vec<Box<i32>> = Vec::new();
    //~^ vec_box
    static S: Vec<Box<i32>> = Vec::new();
    //~^ vec_box

    struct StructWithVecBox {
        sized_type: Vec<Box<SizedStruct>>,
        //~^ vec_box
    }

    struct A(Vec<Box<SizedStruct>>);
    //~^ vec_box
    struct B(Vec<Vec<Box<(u32)>>>);
    //~^ vec_box

    fn allocator_global_defined_vec() -> Vec<Box<i32>, std::alloc::Global> {
        //~^ vec_box
        Vec::new()
    }
    fn allocator_global_defined_box() -> Vec<Box<i32, std::alloc::Global>> {
        //~^ vec_box
        Vec::new()
    }
    fn allocator_match() -> Vec<Box<i32, DummyAllocator>, DummyAllocator> {
        //~^ vec_box
        Vec::new_in(DummyAllocator)
    }
}

/// The following should not trigger the lint
mod should_not_trigger {
    use super::{BigStruct, DummyAllocator, UnsizedStruct};

    struct C(Vec<Box<UnsizedStruct>>);
    struct D(Vec<Box<BigStruct>>);

    struct StructWithVecBoxButItsUnsized {
        unsized_type: Vec<Box<UnsizedStruct>>,
    }

    struct TraitVec<T: ?Sized> {
        // Regression test for #3720. This was causing an ICE.
        inner: Vec<Box<T>>,
    }

    fn allocator_mismatch() -> Vec<Box<i32, DummyAllocator>> {
        Vec::new()
    }
    fn allocator_mismatch_2() -> Vec<Box<i32>, DummyAllocator> {
        Vec::new_in(DummyAllocator)
    }
}

mod inner_mod {
    mod inner {
        pub struct S;
    }

    mod inner2 {
        use super::inner::S;

        pub fn f() -> Vec<Box<S>> {
            //~^ vec_box
            vec![]
        }
    }
}

// https://github.com/rust-lang/rust-clippy/issues/11417
fn in_closure() {
    let _ = |_: Vec<Box<dyn ToString>>| {};
}

fn main() {}
