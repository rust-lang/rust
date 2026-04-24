//! Intrinsics for GPU targets.
//!
//! Intrinsics in this module are intended for use on GPU targets.
//! They can be target specific but in general GPU targets are similar.

#![unstable(feature = "gpu_intrinsics", issue = "none")]

/// Returns the pointer to workgroup memory allocated at launch-time on GPUs.
///
/// Workgroup memory is a memory region that is shared between all threads in
/// the same workgroup. It is faster to access than other memory but pointers do not
/// work outside the workgroup where they were obtained.
/// Workgroup memory can be allocated statically or after compilation, when
/// launching a gpu-kernel. `gpu_launch_sized_workgroup_mem` returns the pointer to
/// the memory that is allocated at launch-time.
/// The size of this memory can differ between launches of a gpu-kernel, depending on
/// what is specified at launch-time.
/// However, the alignment is fixed by the kernel itself, at compile-time.
///
/// The returned pointer is the start of the workgroup memory region that is
/// allocated at launch-time.
/// All calls to `gpu_launch_sized_workgroup_mem` in a workgroup, independent of the
/// generic type, return the same address, so alias the same memory.
/// The returned pointer is aligned by at least the alignment of `T`.
///
/// If `gpu_launch_sized_workgroup_mem` is invoked multiple times with different
/// types that have different alignment, then you may only rely on the resulting
/// pointer having the alignment of `T` after a call to `gpu_launch_sized_workgroup_mem::<T>`
/// has occurred in the current program execution.
///
/// # Safety
///
/// The pointer is safe to dereference from the start (the returned pointer) up to the
/// size of workgroup memory that was specified when launching the current gpu-kernel.
/// This allocated size is not related in any way to `T`.
///
/// The user must take care of synchronizing access to workgroup memory between
/// threads in a workgroup. The usual data race requirements apply.
///
/// # Other APIs
///
/// CUDA and HIP call this dynamic shared memory, shared between threads in a block.
/// OpenCL and SYCL call this local memory, shared between threads in a work-group.
/// GLSL calls this shared memory, shared between invocations in a work group.
/// DirectX calls this groupshared memory, shared between threads in a thread-group.
#[must_use = "returns a pointer that does nothing unless used"]
#[rustc_intrinsic]
#[rustc_nounwind]
#[unstable(feature = "gpu_launch_sized_workgroup_mem", issue = "135513")]
#[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
pub fn gpu_launch_sized_workgroup_mem<T>() -> *mut T;

/// Returns a pointer to the HSA kernel dispatch packet.
///
/// A `gpu-kernel` on amdgpu is always launched through a kernel dispatch packet.
/// The dispatch packet contains the workgroup size, launch size and other data.
/// The content is defined by the [HSA Platform System Architecture Specification],
/// which is implemented e.g. in AMD's [hsa.h].
/// The intrinsic returns a unit pointer so that rustc does not need to know the packet struct.
/// The pointer is valid for the whole lifetime of the program.
///
/// [HSA Platform System Architecture Specification]: https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
/// [hsa.h]: https://github.com/ROCm/rocm-systems/blob/rocm-7.1.0/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h#L2959
#[rustc_nounwind]
#[rustc_intrinsic]
#[cfg(target_arch = "amdgpu")]
#[must_use = "returns a pointer that does nothing unless used"]
pub fn amdgpu_dispatch_ptr() -> *const ();
