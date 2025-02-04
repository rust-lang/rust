//@ run-pass

#![feature(repr_simd, intrinsics)]


#[rustc_intrinsic]
unsafe fn simd_cast_ptr<T, U>(x: T) -> U;

#[rustc_intrinsic]
unsafe fn simd_expose_provenance<T, U>(x: T) -> U;

#[rustc_intrinsic]
unsafe fn simd_with_exposed_provenance<T, U>(x: T) -> U;

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 2]);

fn main() {
    unsafe {
        let mut foo = 4i8;
        let ptr = &mut foo as *mut i8;

        let ptrs = V::<*mut i8>([ptr, core::ptr::null_mut()]);

        // change constness and type
        let const_ptrs: V<*const u8> = simd_cast_ptr(ptrs);

        let exposed_addr: V<usize> = simd_expose_provenance(const_ptrs);

        let with_exposed_provenance: V<*mut i8> = simd_with_exposed_provenance(exposed_addr);

        assert!(const_ptrs.0 == [ptr as *const u8, core::ptr::null()]);
        assert!(exposed_addr.0 == [ptr as usize, 0]);
        assert!(with_exposed_provenance.0 == ptrs.0);
    }
}
