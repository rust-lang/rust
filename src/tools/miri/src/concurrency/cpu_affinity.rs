use crate::bug;
use rustc_target::abi::Endian;

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

    pub fn new(target: &rustc_target::spec::Target, cpu_count: u32) -> Self {
        let mut this = Self([0; Self::CPU_MASK_BYTES]);

        // the default affinity mask includes only the available CPUs
        for i in 0..cpu_count as usize {
            this.set(target, i);
        }

        this
    }

    pub fn chunk_size(target: &rustc_target::spec::Target) -> u64 {
        // The actual representation of the CpuAffinityMask is [c_ulong; _], in practice either
        //
        // - [u32; 32] on 32-bit platforms
        // - [u64; 16] everywhere else

        // FIXME: this should be `size_of::<core::ffi::c_ulong>()`
        u64::from(target.pointer_width / 8)
    }

    fn set(&mut self, target: &rustc_target::spec::Target, cpu: usize) {
        // we silently ignore CPUs that are out of bounds. This matches the behavior of
        // `sched_setaffinity` with a mask that specifies more than `CPU_SETSIZE` CPUs.
        if cpu >= MAX_CPUS {
            return;
        }

        // The actual representation of the CpuAffinityMask is [c_ulong; _], in practice either
        //
        // - [u32; 32] on 32-bit platforms
        // - [u64; 16] everywhere else
        //
        // Within the array elements, we need to use the endianness of the target.
        match Self::chunk_size(target) {
            4 => {
                let start = cpu / 32 * 4; // first byte of the correct u32
                let chunk = self.0[start..].first_chunk_mut::<4>().unwrap();
                let offset = cpu % 32;
                *chunk = match target.options.endian {
                    Endian::Little => (u32::from_le_bytes(*chunk) | 1 << offset).to_le_bytes(),
                    Endian::Big => (u32::from_be_bytes(*chunk) | 1 << offset).to_be_bytes(),
                };
            }
            8 => {
                let start = cpu / 64 * 8; // first byte of the correct u64
                let chunk = self.0[start..].first_chunk_mut::<8>().unwrap();
                let offset = cpu % 64;
                *chunk = match target.options.endian {
                    Endian::Little => (u64::from_le_bytes(*chunk) | 1 << offset).to_le_bytes(),
                    Endian::Big => (u64::from_be_bytes(*chunk) | 1 << offset).to_be_bytes(),
                };
            }
            other => bug!("other chunk sizes are not supported: {other}"),
        };
    }

    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }

    pub fn from_array(
        target: &rustc_target::spec::Target,
        cpu_count: u32,
        bytes: [u8; Self::CPU_MASK_BYTES],
    ) -> Option<Self> {
        // mask by what CPUs are actually available
        let default = Self::new(target, cpu_count);
        let masked = std::array::from_fn(|i| bytes[i] & default.0[i]);

        // at least one thread must be set for the input to be valid
        masked.iter().any(|b| *b != 0).then_some(Self(masked))
    }
}
