use rustc_abi::{Endian, Size};
use rustc_middle::ty::layout::LayoutOf;
use rustc_target::spec::Os;

use crate::*;

/// The maximum number of CPUs supported by miri.
///
/// This value is compatible with the libc `CPU_SETSIZE` constant and corresponds to the number
/// of CPUs that a `cpu_set_t` can contain.
///
/// Real machines can have more CPUs than this number, and there exist APIs to set their affinity,
/// but this is not currently supported by miri.
pub const MAX_CPUS: usize = 1024;

/// A thread's CPU affinity mask determines the set of CPUs on which it is eligible to run.
// the actual representation depends on the target's endianness and pointer width.
// See CpuAffinityMask::set for details
#[derive(Clone)]
pub struct CpuAffinityMask([u8; Self::CPU_MASK_BYTES]);

impl CpuAffinityMask {
    pub(crate) const CPU_MASK_BYTES: usize = MAX_CPUS / 8;

    pub fn new<'tcx>(cx: &impl LayoutOf<'tcx>, cpu_count: u32) -> Self {
        let mut this = Self([0; Self::CPU_MASK_BYTES]);

        // the default affinity mask includes only the available CPUs
        for i in 0..cpu_count.to_usize() {
            this.set(cx, i);
        }

        this
    }

    pub fn chunk_size<'tcx>(cx: &impl LayoutOf<'tcx>) -> u64 {
        // The actual representation of the CpuAffinityMask is [c_ulong; _].
        let ulong = helpers::path_ty_layout(cx, &["core", "ffi", "c_ulong"]);
        ulong.size.bytes()
    }

    fn set<'tcx>(&mut self, cx: &impl LayoutOf<'tcx>, cpu: usize) {
        // we silently ignore CPUs that are out of bounds. This matches the behavior of
        // `sched_setaffinity` with a mask that specifies more than `CPU_SETSIZE` CPUs.
        if cpu >= MAX_CPUS {
            return;
        }

        // The actual representation of the CpuAffinityMask is [c_ulong; _].
        // Within the array elements, we need to use the endianness of the target.
        let target = &cx.tcx().sess.target;
        #[expect(clippy::arithmetic_side_effects)] // we checked above that `cpu` is small enough
        match Self::chunk_size(cx) {
            4 => {
                let start = cpu / 32 * 4; // first byte of the correct u32
                let chunk = self.0[start..].first_chunk_mut::<4>().unwrap();
                let offset = cpu % 32;
                *chunk = match target.options.endian {
                    Endian::Little => (u32::from_le_bytes(*chunk) | (1 << offset)).to_le_bytes(),
                    Endian::Big => (u32::from_be_bytes(*chunk) | (1 << offset)).to_be_bytes(),
                };
            }
            8 => {
                let start = cpu / 64 * 8; // first byte of the correct u64
                let chunk = self.0[start..].first_chunk_mut::<8>().unwrap();
                let offset = cpu % 64;
                *chunk = match target.options.endian {
                    Endian::Little => (u64::from_le_bytes(*chunk) | (1 << offset)).to_le_bytes(),
                    Endian::Big => (u64::from_be_bytes(*chunk) | (1 << offset)).to_be_bytes(),
                };
            }
            other => bug!("chunk size not supported: {other}"),
        };
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    pub fn from_array<'tcx>(
        cx: &impl LayoutOf<'tcx>,
        cpu_count: u32,
        bytes: [u8; Self::CPU_MASK_BYTES],
    ) -> Option<Self> {
        // mask by what CPUs are actually available
        let default = Self::new(cx, cpu_count);
        let masked = std::array::from_fn(|i| bytes[i] & default.0[i]);

        // at least one thread must be set for the input to be valid
        masked.iter().any(|b| *b != 0).then_some(Self(masked))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn sched_getaffinity(
        &mut self,
        pid: &OpTy<'tcx>,
        cpusetsize: &OpTy<'tcx>,
        mask: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let pid = this.read_scalar(pid)?.to_u32()?;
        let cpusetsize = this.read_target_usize(cpusetsize)?;
        let mask = this.read_pointer(mask)?;

        if this.machine.thread_cpu_affinity.is_none() {
            throw_unsup_format!("`sched_getaffinity` is not supported on #![no_core] programs")
        }

        let thread_id = if pid == 0 {
            this.active_thread()
        } else if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android) {
            // On Linux/Android, pid can be a TID as returned by `gettid`.
            let Some(thread_id) = this.get_thread_id_from_linux_tid(pid) else {
                this.set_errno_and_return_neg1(LibcError("ESRCH"), dest)?;
                return interp_ok(());
            };
            thread_id
        } else {
            throw_unsup_format!(
                "`sched_getaffinity` is only supported with a pid of 0 (indicating the current thread) on non-Linux platforms"
            )
        };

        // The mask is stored in chunks, and the size must be a whole number of chunks.
        let chunk_size = CpuAffinityMask::chunk_size(this);

        if this.ptr_is_null(mask)? {
            this.set_errno_and_return_neg1(LibcError("EFAULT"), dest)?;
        } else if cpusetsize == 0 || cpusetsize.checked_rem(chunk_size).unwrap() != 0 {
            // we only copy whole chunks of size_of::<c_ulong>()
            this.set_errno_and_return_neg1(LibcError("EINVAL"), dest)?;
        } else if let Some(cpuset) =
            this.machine.thread_cpu_affinity.as_ref().unwrap().get(&thread_id)
        {
            let cpuset = cpuset.clone();
            // we only copy whole chunks of size_of::<c_ulong>()
            let byte_count = Ord::min(cpuset.as_slice().len(), cpusetsize.try_into().unwrap());
            this.write_bytes_ptr(mask, cpuset.as_slice()[..byte_count].iter().copied())?;
            this.write_null(dest)?;
        } else {
            unreachable!("we validated the thread ID above");
        }

        interp_ok(())
    }

    fn sched_setaffinity(
        &mut self,
        pid: &OpTy<'tcx>,
        cpusetsize: &OpTy<'tcx>,
        mask: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let pid = this.read_scalar(pid)?.to_u32()?;
        let cpusetsize = this.read_target_usize(cpusetsize)?;
        let mask = this.read_pointer(mask)?;

        if this.machine.thread_cpu_affinity.is_none() {
            throw_unsup_format!("`sched_setaffinity` is not supported on #![no_core] programs")
        }

        let thread_id = if pid == 0 {
            this.active_thread()
        } else if matches!(this.tcx.sess.target.os, Os::Linux | Os::Android) {
            // On Linux/Android, pid can be a TID as returned by `gettid`.
            let Some(thread_id) = this.get_thread_id_from_linux_tid(pid) else {
                this.set_errno_and_return_neg1(LibcError("ESRCH"), dest)?;
                return interp_ok(());
            };
            thread_id
        } else {
            throw_unsup_format!(
                "`sched_setaffinity` is only supported with a pid of 0 (indicating the current thread) on non-Linux platforms"
            )
        };

        if this.ptr_is_null(mask)? {
            this.set_errno_and_return_neg1(LibcError("EFAULT"), dest)?;
        } else {
            // NOTE: cpusetsize might be smaller than `CpuAffinityMask::CPU_MASK_BYTES`.
            // Any unspecified bytes are treated as zero here (none of the CPUs are configured).
            // This is not exactly documented, so we assume that this is the behavior in practice.
            let bits_slice =
                this.read_bytes_ptr_strip_provenance(mask, Size::from_bytes(cpusetsize))?;
            // This ignores the bytes beyond `CpuAffinityMask::CPU_MASK_BYTES`
            let bits_array: [u8; CpuAffinityMask::CPU_MASK_BYTES] =
                std::array::from_fn(|i| bits_slice.get(i).copied().unwrap_or(0));
            match CpuAffinityMask::from_array(this, this.machine.num_cpus, bits_array) {
                Some(cpuset) => {
                    this.machine.thread_cpu_affinity.as_mut().unwrap().insert(thread_id, cpuset);
                    this.write_null(dest)?;
                }
                None => {
                    // The intersection between the mask and the available CPUs was empty.
                    this.set_errno_and_return_neg1(LibcError("EINVAL"), dest)?;
                }
            }
        }

        interp_ok(())
    }
}
