//! Checks the basic usage pointers into the non-generic address space.

//@ add-minicore
//@ compile-flags: --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ needs-llvm-components: amdgpu

#![crate_type = "lib"]
#![no_core]
#![feature(abi_unadjusted, intrinsics, lang_items, link_llvm_intrinsics, no_core, rustc_attrs)]

extern crate minicore;
use minicore::*;

#[rustc_intrinsic]
const unsafe fn size_of_val<T>(x: *const T) -> usize;

mod addrspace {
    pub const GENERIC: u32 = 0;
    pub const WORKGROUP: u32 = 3;
    pub const CONST: u32 = 4;
}

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_cast<
    T: 'static,
    U: 'static,
    const SOURCE_ADDRSPACE: u32,
    const TARGET_ADDRSPACE: u32,
>(
    ptr: AddrspacePtr<T, SOURCE_ADDRSPACE>,
) -> AddrspacePtr<U, TARGET_ADDRSPACE>;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_from_ptr<T: 'static>(
    ptr: *mut T,
) -> AddrspacePtr<T, { addrspace::GENERIC }>;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_to_ptr<T: 'static>(ptr: AddrspacePtr<T, { addrspace::GENERIC }>) -> *mut T;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_to_addr<T, const ADDRSPACE: u32>(ptr: AddrspacePtr<(), ADDRSPACE>) -> T;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_offset<T: 'static, const ADDRSPACE: u32>(
    ptr: AddrspacePtr<T, ADDRSPACE>,
    count: isize,
) -> AddrspacePtr<T, ADDRSPACE>;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_arith_offset<T: 'static, const ADDRSPACE: u32>(
    ptr: AddrspacePtr<T, ADDRSPACE>,
    count: isize,
) -> AddrspacePtr<T, ADDRSPACE>;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_read_via_copy<T: 'static, const ADDRSPACE: u32>(
    ptr: AddrspacePtr<T, ADDRSPACE>,
) -> T;

#[rustc_intrinsic]
#[rustc_nounwind]
unsafe fn addrspace_ptr_write_via_move<T: 'static, const ADDRSPACE: u32>(
    ptr: AddrspacePtr<T, ADDRSPACE>,
    value: T,
);

#[repr(C)]
struct DispatchPacket {
    header: u16,
}

extern "unadjusted" {
    #[link_name = "llvm.amdgcn.dispatch.ptr"]
    fn dispatch_ptr() -> AddrspacePtr<DispatchPacket, { addrspace::CONST }>;
}

#[lang = "addrspace_ptr_type"]
pub struct AddrspacePtr<T: 'static, const ADDRSPACE: u32> {
    // Struct implementation is replaced by compiler.
    // This field is here for using the generic arguments but cannot be set or used in any way.
    do_not_use: PhantomData<*const T>,
}

impl<T: 'static, const ADDRSPACE: u32> AddrspacePtr<T, ADDRSPACE> {
    unsafe fn addr<U>(self) -> U {
        unsafe { addrspace_ptr_to_addr(self.cast::<()>()) }
    }

    unsafe fn cast<U>(self) -> AddrspacePtr<U, ADDRSPACE> {
        unsafe { addrspace_ptr_cast(self) }
    }

    unsafe fn cast_addrspace<const TARGET_ADDRSPACE: u32>(
        self,
    ) -> AddrspacePtr<T, TARGET_ADDRSPACE> {
        unsafe { addrspace_ptr_cast(self) }
    }

    unsafe fn offset(self, count: isize) -> Self {
        unsafe { addrspace_ptr_offset(self, count) }
    }

    unsafe fn wrapping_offset(self, count: isize) -> Self {
        unsafe { addrspace_ptr_arith_offset(self, count) }
    }

    unsafe fn read(self) -> T {
        unsafe { addrspace_ptr_read_via_copy(self) }
    }

    unsafe fn write(self, value: T) {
        unsafe { addrspace_ptr_write_via_move(self, value) }
    }
}

impl<T: 'static> AddrspacePtr<T, { addrspace::GENERIC }> {
    unsafe fn from_ptr(ptr: *mut T) -> Self {
        unsafe { addrspace_ptr_from_ptr(ptr) }
    }

    unsafe fn as_ptr(self) -> *mut T {
        unsafe { addrspace_ptr_to_ptr(self) }
    }
}

#[unsafe(no_mangle)]
fn addr(ptr: AddrspacePtr<u8, { addrspace::WORKGROUP }>) -> u32 {
    // CHECK-LABEL: @addr
    // CHECK: %[[val:[^ ]+]] = ptrtoint ptr addrspace(3) %ptr to i32
    // CHECK: ret i32 %[[val]]
    unsafe { ptr.addr() }
}

#[unsafe(no_mangle)]
fn cast(
    ptr: AddrspacePtr<u8, { addrspace::WORKGROUP }>,
) -> AddrspacePtr<u32, { addrspace::WORKGROUP }> {
    // CHECK-LABEL: @cast
    // CHECK: ret ptr addrspace(3) %ptr
    unsafe { ptr.cast() }
}

#[unsafe(no_mangle)]
fn cast_addrspace(
    ptr: AddrspacePtr<u8, { addrspace::WORKGROUP }>,
) -> AddrspacePtr<u8, { addrspace::GENERIC }> {
    // CHECK-LABEL: @cast_addrspace
    // CHECK: %[[val:[^ ]+]] = addrspacecast ptr addrspace(3) %ptr to ptr
    // CHECK: ret ptr %[[val]]
    unsafe { ptr.cast_addrspace() }
}

#[unsafe(no_mangle)]
fn offset(
    ptr: AddrspacePtr<u8, { addrspace::WORKGROUP }>,
) -> AddrspacePtr<u8, { addrspace::WORKGROUP }> {
    // CHECK-LABEL: @offset
    // CHECK: %[[val:[^ ]+]] = getelementptr inbounds i8, ptr addrspace(3) %ptr, i32 -20
    // CHECK: ret ptr addrspace(3) %[[val]]
    unsafe { ptr.offset(-20) }
}

#[unsafe(no_mangle)]
fn wrapping_offset(
    ptr: AddrspacePtr<u8, { addrspace::WORKGROUP }>,
) -> AddrspacePtr<u8, { addrspace::WORKGROUP }> {
    // CHECK-LABEL: @wrapping_offset
    // CHECK: %[[val:[^ ]+]] = getelementptr i8, ptr addrspace(3) %ptr, i32 -15
    // CHECK: ret ptr addrspace(3) %[[val]]
    unsafe { ptr.wrapping_offset(-15) }
}

#[unsafe(no_mangle)]
fn read(ptr: AddrspacePtr<u32, { addrspace::WORKGROUP }>) -> u32 {
    // CHECK-LABEL: @read
    // CHECK: %[[val:[^ ]+]] = load i32, ptr addrspace(3) %ptr, align 4
    // CHECK: ret i32 %[[val]]
    unsafe { ptr.read() }
}

#[unsafe(no_mangle)]
fn write(ptr: AddrspacePtr<u32, { addrspace::WORKGROUP }>, val: u32) {
    // CHECK-LABEL: @write
    // CHECK: store i32 %val, ptr addrspace(3) %ptr, align 4
    // CHECK: ret void
    unsafe { ptr.write(val) }
}

#[unsafe(no_mangle)]
fn fun(ptr: *mut u8) -> (AddrspacePtr<u8, { addrspace::WORKGROUP }>, usize) {
    // CHECK-LABEL: @fun
    // CHECK: %[[v0:[^ ]+]] = addrspacecast ptr %ptr to ptr addrspace(3)
    // CHECK: %[[v1:[^ ]+]] = insertvalue { ptr addrspace(3), i64 } poison, ptr addrspace(3) %[[v0]], 0
    // CHECK: %[[res:[^ ]+]] = insertvalue { ptr addrspace(3), i64 } %[[v1]], i64 4, 1
    // CHECK: ret { ptr addrspace(3), i64 } %[[res]]

    unsafe {
        let p: AddrspacePtr<u8, { addrspace::WORKGROUP }> =
            AddrspacePtr::from_ptr(ptr).cast_addrspace();
        let size = size_of_val(&p);
        (p, size)
    }
}

#[unsafe(no_mangle)]
fn get_raw_dispatch_ptr() -> AddrspacePtr<DispatchPacket, { addrspace::CONST }> {
    // CHECK-LABEL: @get_raw_dispatch_ptr
    // CHECK: %[[val:[^ ]+]] = tail call noundef ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
    // CHECK: ret ptr addrspace(4) %[[val]]
    unsafe { dispatch_ptr() }
}

#[unsafe(no_mangle)]
fn get_dispatch_ptr() -> &'static DispatchPacket {
    // CHECK-LABEL: @get_dispatch_ptr
    // CHECK: %[[val:[^ ]+]] = tail call noundef ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
    // CHECK: %[[res:[^ ]+]] = addrspacecast ptr addrspace(4) %[[val]] to ptr
    // CHECK: ret ptr %[[res]]
    unsafe { &*dispatch_ptr().cast_addrspace::<{ addrspace::GENERIC }>().as_ptr() }
}

#[unsafe(no_mangle)]
fn read_ptr(
    ptr: AddrspacePtr<u8, { addrspace::CONST }>,
) -> AddrspacePtr<u8, { addrspace::WORKGROUP }> {
    // Read a pointer to a specific addrspace from a pointer
    // CHECK-LABEL: @read
    // CHECK: %[[val:[^ ]+]] = load ptr addrspace(3), ptr addrspace(4) %ptr, align 4
    // CHECK: ret ptr addrspace(3) %[[val]]
    unsafe { ptr.cast::<AddrspacePtr<u8, { addrspace::WORKGROUP }>>().read() }
}

const EXPECTED: usize = 4;
const ACTUAL: usize = mem::size_of::<AddrspacePtr<u8, { addrspace::WORKGROUP }>>();
// Validate that the size is 4 byte, which is different from the generic pointer size
const _: [(); EXPECTED] = [(); ACTUAL];
