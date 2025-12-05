use rustc_abi::Endian;
use rustc_middle::ty::layout::LayoutOf;

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
pub(crate) struct CpuAffinityMask([u8; Self::CPU_MASK_BYTES]);

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
