//@ compile-flags: -C opt-level=3
#![crate_type = "lib"]

mod foobar {
    use std::alloc::{GlobalAlloc, Layout};

    struct Allocator;

    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // CHECK-LABEL: ; __rustc::__rust_alloc
            // CHECK-NEXT: ; Function Attrs: {{.*}}allockind("alloc,uninitialized,aligned") allocsize(0){{.*}}
            // CHECK-NEXT: define{{.*}} noalias{{.*}} ptr @{{.*}}__rust_alloc(i[[SIZE:[0-9]+]] {{.*}}%size, i[[SIZE]] allocalign{{.*}} %align)
            panic!()
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // CHECK-LABEL: ; __rustc::__rust_dealloc
            // CHECK-NEXT: ; Function Attrs: {{.*}}allockind("free"){{.*}}
            // CHECK-NEXT: define{{.*}} void @{{.*}}__rust_dealloc(ptr allocptr{{.*}} %ptr, i[[SIZE]] {{.*}} %size, i[[SIZE]] {{.*}} %align)
            panic!()
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // CHECK-LABEL: ; __rustc::__rust_realloc
            // CHECK-NEXT: ; Function Attrs: {{.*}}allockind("realloc,aligned") allocsize(3){{.*}}
            // CHECK-NEXT: define{{.*}} noalias{{.*}} ptr @{{.*}}__rust_realloc(ptr allocptr{{.*}} %ptr, i[[SIZE]] {{.*}} %size, i[[SIZE]] allocalign{{.*}} %align, i[[SIZE]] {{.*}} %new_size)
            panic!()
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // CHECK-LABEL: ; __rustc::__rust_alloc_zeroed
            // CHECK-NEXT: ; Function Attrs: {{.*}}allockind("alloc,zeroed,aligned") allocsize(0){{.*}}
            // CHECK-NEXT: define{{.*}} noalias{{.*}} ptr @{{.*}}__rust_alloc_zeroed(i[[SIZE]] {{.*}} %size, i[[SIZE]] allocalign{{.*}} %align)
            panic!()
        }
    }

    #[global_allocator]
    static GLOBAL: Allocator = Allocator;
}
