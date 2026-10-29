//! Run-time feature detection for RISC-V on Linux.
//!
//! On RISC-V, detection using auxv only supports single-letter extensions.
//! So, we use riscv_hwprobe that supports multi-letter extensions if available.
//! <https://www.kernel.org/doc/html/latest/arch/riscv/hwprobe.html>

use core::ptr;

use super::super::riscv::imply_features;
use super::auxvec;
use crate::detect::{Feature, bit, cache};

// See <https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/prctl.h?h=v6.16>
// for runtime status query constants.
const PR_RISCV_V_GET_CONTROL: libc::c_int = 70;
const PR_RISCV_V_VSTATE_CTRL_ON: libc::c_int = 2;
const PR_RISCV_V_VSTATE_CTRL_CUR_MASK: libc::c_int = 3;

// See <https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/riscv/include/uapi/asm/hwprobe.h?h=v6.16>
// for riscv_hwprobe struct and hardware probing constants.

#[repr(C)]
struct riscv_hwprobe {
    key: i64,
    value: u64,
}

impl riscv_hwprobe {
    // key is overwritten to -1 if not supported by riscv_hwprobe syscall.
    pub fn get(&self) -> Option<u64> {
        (self.key != -1).then_some(self.value)
    }
}

#[allow(non_upper_case_globals)]
const __NR_riscv_hwprobe: libc::c_long = 258;

const RISCV_HWPROBE_KEY_BASE_BEHAVIOR: i64 = 3;
const RISCV_HWPROBE_BASE_BEHAVIOR_IMA: u64 = 1 << 0;

const RISCV_HWPROBE_KEY_IMA_EXT_0: i64 = 4;
const RISCV_HWPROBE_IMA_FD: u64 = 1 << 0;
const RISCV_HWPROBE_IMA_C: u64 = 1 << 1;
const RISCV_HWPROBE_IMA_V: u64 = 1 << 2;
const RISCV_HWPROBE_EXT_ZBA: u64 = 1 << 3;
const RISCV_HWPROBE_EXT_ZBB: u64 = 1 << 4;
const RISCV_HWPROBE_EXT_ZBS: u64 = 1 << 5;
const RISCV_HWPROBE_EXT_ZICBOZ: u64 = 1 << 6;
const RISCV_HWPROBE_EXT_ZBC: u64 = 1 << 7;
const RISCV_HWPROBE_EXT_ZBKB: u64 = 1 << 8;
const RISCV_HWPROBE_EXT_ZBKC: u64 = 1 << 9;
const RISCV_HWPROBE_EXT_ZBKX: u64 = 1 << 10;
const RISCV_HWPROBE_EXT_ZKND: u64 = 1 << 11;
const RISCV_HWPROBE_EXT_ZKNE: u64 = 1 << 12;
const RISCV_HWPROBE_EXT_ZKNH: u64 = 1 << 13;
const RISCV_HWPROBE_EXT_ZKSED: u64 = 1 << 14;
const RISCV_HWPROBE_EXT_ZKSH: u64 = 1 << 15;
const RISCV_HWPROBE_EXT_ZKT: u64 = 1 << 16;
const RISCV_HWPROBE_EXT_ZVBB: u64 = 1 << 17;
const RISCV_HWPROBE_EXT_ZVBC: u64 = 1 << 18;
const RISCV_HWPROBE_EXT_ZVKB: u64 = 1 << 19;
const RISCV_HWPROBE_EXT_ZVKG: u64 = 1 << 20;
const RISCV_HWPROBE_EXT_ZVKNED: u64 = 1 << 21;
const RISCV_HWPROBE_EXT_ZVKNHA: u64 = 1 << 22;
const RISCV_HWPROBE_EXT_ZVKNHB: u64 = 1 << 23;
const RISCV_HWPROBE_EXT_ZVKSED: u64 = 1 << 24;
const RISCV_HWPROBE_EXT_ZVKSH: u64 = 1 << 25;
const RISCV_HWPROBE_EXT_ZVKT: u64 = 1 << 26;
const RISCV_HWPROBE_EXT_ZFH: u64 = 1 << 27;
const RISCV_HWPROBE_EXT_ZFHMIN: u64 = 1 << 28;
const RISCV_HWPROBE_EXT_ZIHINTNTL: u64 = 1 << 29;
const RISCV_HWPROBE_EXT_ZVFH: u64 = 1 << 30;
const RISCV_HWPROBE_EXT_ZVFHMIN: u64 = 1 << 31;
const RISCV_HWPROBE_EXT_ZFA: u64 = 1 << 32;
const RISCV_HWPROBE_EXT_ZTSO: u64 = 1 << 33;
const RISCV_HWPROBE_EXT_ZACAS: u64 = 1 << 34;
const RISCV_HWPROBE_EXT_ZICOND: u64 = 1 << 35;
const RISCV_HWPROBE_EXT_ZIHINTPAUSE: u64 = 1 << 36;
const RISCV_HWPROBE_EXT_ZVE32X: u64 = 1 << 37;
const RISCV_HWPROBE_EXT_ZVE32F: u64 = 1 << 38;
const RISCV_HWPROBE_EXT_ZVE64X: u64 = 1 << 39;
const RISCV_HWPROBE_EXT_ZVE64F: u64 = 1 << 40;
const RISCV_HWPROBE_EXT_ZVE64D: u64 = 1 << 41;
const RISCV_HWPROBE_EXT_ZIMOP: u64 = 1 << 42;
const RISCV_HWPROBE_EXT_ZCA: u64 = 1 << 43;
const RISCV_HWPROBE_EXT_ZCB: u64 = 1 << 44;
const RISCV_HWPROBE_EXT_ZCD: u64 = 1 << 45;
const RISCV_HWPROBE_EXT_ZCF: u64 = 1 << 46;
const RISCV_HWPROBE_EXT_ZCMOP: u64 = 1 << 47;
const RISCV_HWPROBE_EXT_ZAWRS: u64 = 1 << 48;
// Excluded because it only reports the existence of `prctl`-based pointer masking control.
// const RISCV_HWPROBE_EXT_SUPM: u64 = 1 << 49;
const RISCV_HWPROBE_EXT_ZICNTR: u64 = 1 << 50;
const RISCV_HWPROBE_EXT_ZIHPM: u64 = 1 << 51;
const RISCV_HWPROBE_EXT_ZFBFMIN: u64 = 1 << 52;
const RISCV_HWPROBE_EXT_ZVFBFMIN: u64 = 1 << 53;
const RISCV_HWPROBE_EXT_ZVFBFWMA: u64 = 1 << 54;
const RISCV_HWPROBE_EXT_ZICBOM: u64 = 1 << 55;
const RISCV_HWPROBE_EXT_ZAAMO: u64 = 1 << 56;
const RISCV_HWPROBE_EXT_ZALRSC: u64 = 1 << 57;
const RISCV_HWPROBE_EXT_ZABHA: u64 = 1 << 58;

const RISCV_HWPROBE_KEY_CPUPERF_0: i64 = 5;
const RISCV_HWPROBE_MISALIGNED_FAST: u64 = 3;
const RISCV_HWPROBE_MISALIGNED_MASK: u64 = 7;

const RISCV_HWPROBE_KEY_MISALIGNED_SCALAR_PERF: i64 = 9;
const RISCV_HWPROBE_MISALIGNED_SCALAR_FAST: u64 = 3;

const RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF: i64 = 10;
const RISCV_HWPROBE_MISALIGNED_VECTOR_FAST: u64 = 3;

// syscall returns an unsupported error if riscv_hwprobe is not supported,
// so we can safely use this function on older versions of Linux.
fn _riscv_hwprobe(out: &mut [riscv_hwprobe]) -> bool {
    unsafe fn __riscv_hwprobe(
        pairs: *mut riscv_hwprobe,
        pair_count: libc::size_t,
        cpu_set_size: libc::size_t,
        cpus: *mut libc::c_ulong,
        flags: libc::c_uint,
    ) -> libc::c_long {
        unsafe { libc::syscall(__NR_riscv_hwprobe, pairs, pair_count, cpu_set_size, cpus, flags) }
    }

    unsafe { __riscv_hwprobe(out.as_mut_ptr(), out.len(), 0, ptr::null_mut(), 0) == 0 }
}

/// Read list of supported features from (1) the auxiliary vector
/// and (2) the results of `riscv_hwprobe` and `prctl` system calls.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let mut enable_feature = |feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };

    // Use auxiliary vector to enable single-letter ISA extensions.
    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/riscv/include/uapi/asm/hwcap.h?h=v6.16
    let auxv = auxvec::auxv().expect("read auxvec"); // should not fail on RISC-V platform
    let mut has_i = bit::test(auxv.hwcap, (b'i' - b'a').into());
    #[allow(clippy::eq_op)]
    enable_feature(Feature::a, bit::test(auxv.hwcap, (b'a' - b'a').into()));
    enable_feature(Feature::c, bit::test(auxv.hwcap, (b'c' - b'a').into()));
    enable_feature(Feature::d, bit::test(auxv.hwcap, (b'd' - b'a').into()));
    enable_feature(Feature::f, bit::test(auxv.hwcap, (b'f' - b'a').into()));
    enable_feature(Feature::m, bit::test(auxv.hwcap, (b'm' - b'a').into()));
    let has_v = bit::test(auxv.hwcap, (b'v' - b'a').into());
    let mut is_v_set = false;

    // Use riscv_hwprobe syscall to query more extensions and
    // performance-related capabilities.
    'hwprobe: {
        macro_rules! init {
            { $($name: ident : $key: expr),* $(,)? } => {
                #[repr(usize)]
                enum Indices { $($name),* }
                let mut t = [$(riscv_hwprobe { key: $key, value: 0 }),*];
                macro_rules! data_mut { () => { &mut t } }
                macro_rules! query { [$idx: ident] => { t[Indices::$idx as usize].get() } }
            }
        }
        init! {
            BaseBehavior: RISCV_HWPROBE_KEY_BASE_BEHAVIOR,
            Extensions:   RISCV_HWPROBE_KEY_IMA_EXT_0,
            MisalignedScalarPerf: RISCV_HWPROBE_KEY_MISALIGNED_SCALAR_PERF,
            MisalignedVectorPerf: RISCV_HWPROBE_KEY_MISALIGNED_VECTOR_PERF,
            MisalignedScalarPerfFallback: RISCV_HWPROBE_KEY_CPUPERF_0,
        };
        if !_riscv_hwprobe(data_mut!()) {
            break 'hwprobe;
        }

        // Query scalar misaligned behavior.
        if let Some(value) = query![MisalignedScalarPerf] {
            enable_feature(
                Feature::unaligned_scalar_mem,
                value == RISCV_HWPROBE_MISALIGNED_SCALAR_FAST,
            );
        } else if let Some(value) = query![MisalignedScalarPerfFallback] {
            // Deprecated method for fallback
            enable_feature(
                Feature::unaligned_scalar_mem,
                value & RISCV_HWPROBE_MISALIGNED_MASK == RISCV_HWPROBE_MISALIGNED_FAST,
            );
        }

        // Query vector misaligned behavior.
        if let Some(value) = query![MisalignedVectorPerf] {
            enable_feature(
                Feature::unaligned_vector_mem,
                value == RISCV_HWPROBE_MISALIGNED_VECTOR_FAST,
            );
        }

        // Query whether "I" base and extensions "M" and "A" (as in the ISA
        // manual version 2.2) are enabled.  "I" base at that time corresponds
        // to "I", "Zicsr", "Zicntr" and "Zifencei" (as in the ISA manual version
        // 20240411).
        // This is a current requirement of
        // `RISCV_HWPROBE_KEY_IMA_EXT_0`-based tests.
        if query![BaseBehavior].is_none_or(|value| value & RISCV_HWPROBE_BASE_BEHAVIOR_IMA == 0) {
            break 'hwprobe;
        }
        has_i = true;
        enable_feature(Feature::zicsr, true);
        enable_feature(Feature::zicntr, true);
        enable_feature(Feature::zifencei, true);
        enable_feature(Feature::m, true);
        enable_feature(Feature::a, true);

        // Enable features based on `RISCV_HWPROBE_KEY_IMA_EXT_0`.
        let Some(ima_ext_0) = query![Extensions] else {
            break 'hwprobe;
        };
        let test = |mask| (ima_ext_0 & mask) != 0;

        enable_feature(Feature::d, test(RISCV_HWPROBE_IMA_FD)); // F is implied.
        enable_feature(Feature::c, test(RISCV_HWPROBE_IMA_C));

        enable_feature(Feature::zicntr, test(RISCV_HWPROBE_EXT_ZICNTR));
        enable_feature(Feature::zihpm, test(RISCV_HWPROBE_EXT_ZIHPM));

        enable_feature(Feature::zihintntl, test(RISCV_HWPROBE_EXT_ZIHINTNTL));
        enable_feature(Feature::zihintpause, test(RISCV_HWPROBE_EXT_ZIHINTPAUSE));
        enable_feature(Feature::zimop, test(RISCV_HWPROBE_EXT_ZIMOP));
        enable_feature(Feature::zicbom, test(RISCV_HWPROBE_EXT_ZICBOM));
        enable_feature(Feature::zicboz, test(RISCV_HWPROBE_EXT_ZICBOZ));
        enable_feature(Feature::zicond, test(RISCV_HWPROBE_EXT_ZICOND));

        enable_feature(Feature::zalrsc, test(RISCV_HWPROBE_EXT_ZALRSC));
        enable_feature(Feature::zaamo, test(RISCV_HWPROBE_EXT_ZAAMO));
        enable_feature(Feature::zawrs, test(RISCV_HWPROBE_EXT_ZAWRS));
        enable_feature(Feature::zabha, test(RISCV_HWPROBE_EXT_ZABHA));
        enable_feature(Feature::zacas, test(RISCV_HWPROBE_EXT_ZACAS));
        enable_feature(Feature::ztso, test(RISCV_HWPROBE_EXT_ZTSO));

        enable_feature(Feature::zba, test(RISCV_HWPROBE_EXT_ZBA));
        enable_feature(Feature::zbb, test(RISCV_HWPROBE_EXT_ZBB));
        enable_feature(Feature::zbs, test(RISCV_HWPROBE_EXT_ZBS));
        enable_feature(Feature::zbc, test(RISCV_HWPROBE_EXT_ZBC));

        enable_feature(Feature::zbkb, test(RISCV_HWPROBE_EXT_ZBKB));
        enable_feature(Feature::zbkc, test(RISCV_HWPROBE_EXT_ZBKC));
        enable_feature(Feature::zbkx, test(RISCV_HWPROBE_EXT_ZBKX));
        enable_feature(Feature::zknd, test(RISCV_HWPROBE_EXT_ZKND));
        enable_feature(Feature::zkne, test(RISCV_HWPROBE_EXT_ZKNE));
        enable_feature(Feature::zknh, test(RISCV_HWPROBE_EXT_ZKNH));
        enable_feature(Feature::zksed, test(RISCV_HWPROBE_EXT_ZKSED));
        enable_feature(Feature::zksh, test(RISCV_HWPROBE_EXT_ZKSH));
        enable_feature(Feature::zkt, test(RISCV_HWPROBE_EXT_ZKT));

        enable_feature(Feature::zcmop, test(RISCV_HWPROBE_EXT_ZCMOP));
        enable_feature(Feature::zca, test(RISCV_HWPROBE_EXT_ZCA));
        enable_feature(Feature::zcf, test(RISCV_HWPROBE_EXT_ZCF));
        enable_feature(Feature::zcd, test(RISCV_HWPROBE_EXT_ZCD));
        enable_feature(Feature::zcb, test(RISCV_HWPROBE_EXT_ZCB));

        enable_feature(Feature::zfh, test(RISCV_HWPROBE_EXT_ZFH));
        enable_feature(Feature::zfhmin, test(RISCV_HWPROBE_EXT_ZFHMIN));
        enable_feature(Feature::zfa, test(RISCV_HWPROBE_EXT_ZFA));
        enable_feature(Feature::zfbfmin, test(RISCV_HWPROBE_EXT_ZFBFMIN));

        // Use prctl (if any) to determine whether the vector extension
        // is enabled on the current thread (assuming the entire process
        // share the same status).  If prctl fails (e.g. QEMU userland emulator
        // as of version 9.2.3), use auxiliary vector to retrieve the default
        // vector status on the process startup.
        let has_vectors = {
            let v_status = unsafe { libc::prctl(PR_RISCV_V_GET_CONTROL) };
            if v_status >= 0 {
                (v_status & PR_RISCV_V_VSTATE_CTRL_CUR_MASK) == PR_RISCV_V_VSTATE_CTRL_ON
            } else {
                has_v
            }
        };
        if has_vectors {
            enable_feature(Feature::v, test(RISCV_HWPROBE_IMA_V));
            enable_feature(Feature::zve32x, test(RISCV_HWPROBE_EXT_ZVE32X));
            enable_feature(Feature::zve32f, test(RISCV_HWPROBE_EXT_ZVE32F));
            enable_feature(Feature::zve64x, test(RISCV_HWPROBE_EXT_ZVE64X));
            enable_feature(Feature::zve64f, test(RISCV_HWPROBE_EXT_ZVE64F));
            enable_feature(Feature::zve64d, test(RISCV_HWPROBE_EXT_ZVE64D));

            enable_feature(Feature::zvbb, test(RISCV_HWPROBE_EXT_ZVBB));
            enable_feature(Feature::zvbc, test(RISCV_HWPROBE_EXT_ZVBC));
            enable_feature(Feature::zvkb, test(RISCV_HWPROBE_EXT_ZVKB));
            enable_feature(Feature::zvkg, test(RISCV_HWPROBE_EXT_ZVKG));
            enable_feature(Feature::zvkned, test(RISCV_HWPROBE_EXT_ZVKNED));
            enable_feature(Feature::zvknha, test(RISCV_HWPROBE_EXT_ZVKNHA));
            enable_feature(Feature::zvknhb, test(RISCV_HWPROBE_EXT_ZVKNHB));
            enable_feature(Feature::zvksed, test(RISCV_HWPROBE_EXT_ZVKSED));
            enable_feature(Feature::zvksh, test(RISCV_HWPROBE_EXT_ZVKSH));
            enable_feature(Feature::zvkt, test(RISCV_HWPROBE_EXT_ZVKT));

            enable_feature(Feature::zvfh, test(RISCV_HWPROBE_EXT_ZVFH));
            enable_feature(Feature::zvfhmin, test(RISCV_HWPROBE_EXT_ZVFHMIN));
            enable_feature(Feature::zvfbfmin, test(RISCV_HWPROBE_EXT_ZVFBFMIN));
            enable_feature(Feature::zvfbfwma, test(RISCV_HWPROBE_EXT_ZVFBFWMA));
        }
        is_v_set = true;
    };

    // Set V purely depending on the auxiliary vector
    // only if no fine-grained vector extension detection is available.
    if !is_v_set {
        enable_feature(Feature::v, has_v);
    }

    // Handle base ISA.
    // If future RV128I is supported, implement with `enable_feature` here.
    // Note that we should use `target_arch` instead of `target_pointer_width`
    // to avoid misdetection caused by experimental ABIs such as RV64ILP32.
    #[cfg(target_arch = "riscv64")]
    enable_feature(Feature::rv64i, has_i);
    #[cfg(target_arch = "riscv32")]
    enable_feature(Feature::rv32i, has_i);

    imply_features(value)
}
