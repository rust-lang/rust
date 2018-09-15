//! NVPTX intrinsics (experimental)
//!
//! These intrinsics form the foundation of the CUDA
//! programming model.
//!
//! The reference is the [CUDA C Programming Guide][cuda_c]. Relevant is also
//! the [LLVM NVPTX Backend documentation][llvm_docs].
//!
//! [cuda_c]:
//! http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//! [llvm_docs]:
//! https://llvm.org/docs/NVPTXUsage.html

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.nvvm.barrier0"]
    fn syncthreads() -> ();
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    fn block_dim_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.y"]
    fn block_dim_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.z"]
    fn block_dim_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.y"]
    fn block_idx_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.z"]
    fn block_idx_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.x"]
    fn grid_dim_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.y"]
    fn grid_dim_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.z"]
    fn grid_dim_z() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.y"]
    fn thread_idx_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.z"]
    fn thread_idx_z() -> i32;
}

/// Synchronizes all threads in the block.
#[inline]
pub unsafe fn _syncthreads() -> () {
    syncthreads()
}

/// x-th thread-block dimension.
#[inline]
pub unsafe fn _block_dim_x() -> i32 {
    block_dim_x()
}

/// y-th thread-block dimension.
#[inline]
pub unsafe fn _block_dim_y() -> i32 {
    block_dim_y()
}

/// z-th thread-block dimension.
#[inline]
pub unsafe fn _block_dim_z() -> i32 {
    block_dim_z()
}

/// x-th thread-block index.
#[inline]
pub unsafe fn _block_idx_x() -> i32 {
    block_idx_x()
}

/// y-th thread-block index.
#[inline]
pub unsafe fn _block_idx_y() -> i32 {
    block_idx_y()
}

/// z-th thread-block index.
#[inline]
pub unsafe fn _block_idx_z() -> i32 {
    block_idx_z()
}

/// x-th block-grid dimension.
#[inline]
pub unsafe fn _grid_dim_x() -> i32 {
    grid_dim_x()
}

/// y-th block-grid dimension.
#[inline]
pub unsafe fn _grid_dim_y() -> i32 {
    grid_dim_y()
}

/// z-th block-grid dimension.
#[inline]
pub unsafe fn _grid_dim_z() -> i32 {
    grid_dim_z()
}

/// x-th thread index.
#[inline]
pub unsafe fn _thread_idx_x() -> i32 {
    thread_idx_x()
}

/// y-th thread index.
#[inline]
pub unsafe fn _thread_idx_y() -> i32 {
    thread_idx_y()
}

/// z-th thread index.
#[inline]
pub unsafe fn _thread_idx_z() -> i32 {
    thread_idx_z()
}

/// Generates the trap instruction `TRAP`
#[cfg_attr(test, assert_instr(trap))]
#[inline]
pub unsafe fn trap() -> ! {
    ::_core::intrinsics::abort()
}
