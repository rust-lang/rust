//! Run-time feature detection for s390x on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector
pub(crate) fn detect_features() -> cache::Initializer {
    let opt_hwcap: Option<AtHwcap> = auxvec::auxv().ok().map(Into::into);
    let facilities = ExtendedFacilityList::new();
    cache(opt_hwcap, facilities)
}

#[derive(Debug, Default, PartialEq)]
struct AtHwcap {
    esan3: bool,
    zarch: bool,
    stfle: bool,
    msa: bool,
    ldisp: bool,
    eimm: bool,
    dfp: bool,
    hpage: bool,
    etf3eh: bool,
    high_gprs: bool,
    te: bool,
    vxrs: bool,
    vxrs_bcd: bool,
    vxrs_ext: bool,
    gs: bool,
    vxrs_ext2: bool,
    vxrs_pde: bool,
    sort: bool,
    dflt: bool,
    vxrs_pde2: bool,
    nnpa: bool,
    pci_mio: bool,
    sie: bool,
}

impl From<auxvec::AuxVec> for AtHwcap {
    /// Reads AtHwcap from the auxiliary vector.
    fn from(auxv: auxvec::AuxVec) -> Self {
        AtHwcap {
            esan3: bit::test(auxv.hwcap, 0),
            zarch: bit::test(auxv.hwcap, 1),
            stfle: bit::test(auxv.hwcap, 2),
            msa: bit::test(auxv.hwcap, 3),
            ldisp: bit::test(auxv.hwcap, 4),
            eimm: bit::test(auxv.hwcap, 5),
            dfp: bit::test(auxv.hwcap, 6),
            hpage: bit::test(auxv.hwcap, 7),
            etf3eh: bit::test(auxv.hwcap, 8),
            high_gprs: bit::test(auxv.hwcap, 9),
            te: bit::test(auxv.hwcap, 10),
            vxrs: bit::test(auxv.hwcap, 11),
            vxrs_bcd: bit::test(auxv.hwcap, 12),
            vxrs_ext: bit::test(auxv.hwcap, 13),
            gs: bit::test(auxv.hwcap, 14),
            vxrs_ext2: bit::test(auxv.hwcap, 15),
            vxrs_pde: bit::test(auxv.hwcap, 16),
            sort: bit::test(auxv.hwcap, 17),
            dflt: bit::test(auxv.hwcap, 18),
            vxrs_pde2: bit::test(auxv.hwcap, 19),
            nnpa: bit::test(auxv.hwcap, 20),
            pci_mio: bit::test(auxv.hwcap, 21),
            sie: bit::test(auxv.hwcap, 22),
        }
    }
}

struct ExtendedFacilityList([u64; 4]);

impl ExtendedFacilityList {
    fn new() -> Self {
        let mut result: [u64; 4] = [0; 4];
        // SAFETY: rust/llvm only support s390x version with the `stfle` instruction.
        unsafe {
            core::arch::asm!(
                // equivalently ".insn s, 0xb2b00000, 0({1})",
                "stfle 0({})",
                in(reg_addr) result.as_mut_ptr() ,
                inout("r0") result.len() as u64 - 1 => _,
                options(nostack)
            );
        }
        Self(result)
    }

    const fn get_bit(&self, n: usize) -> bool {
        // NOTE: bits are numbered from the left.
        self.0[n / 64] & (1 << (63 - (n % 64))) != 0
    }
}

/// Initializes the cache from the feature bits.
///
/// These values are part of the platform-specific [asm/elf.h][kernel], and are a selection of the
/// fields found in the [Facility Indications].
///
/// [Facility Indications]: https://www.ibm.com/support/pages/sites/default/files/2021-05/SA22-7871-10.pdf#page=63
/// [kernel]: https://github.com/torvalds/linux/blob/b62cef9a5c673f1b8083159f5dc03c1c5daced2f/arch/s390/include/asm/elf.h#L129
fn cache(hwcap: Option<AtHwcap>, facilities: ExtendedFacilityList) -> cache::Initializer {
    let mut value = cache::Initializer::default();

    {
        let mut enable_if_set = |bit_index, f| {
            if facilities.get_bit(bit_index) {
                value.set(f as u32);
            }
        };

        // We use HWCAP for `vector` because it requires both hardware and kernel support.
        if let Some(AtHwcap { vxrs: true, .. }) = hwcap {
            // vector and related

            enable_if_set(129, Feature::vector);

            enable_if_set(135, Feature::vector_enhancements_1);
            enable_if_set(148, Feature::vector_enhancements_2);
            enable_if_set(198, Feature::vector_enhancements_3);

            enable_if_set(134, Feature::vector_packed_decimal);
            enable_if_set(152, Feature::vector_packed_decimal_enhancement);
            enable_if_set(192, Feature::vector_packed_decimal_enhancement_2);
            enable_if_set(199, Feature::vector_packed_decimal_enhancement_3);

            enable_if_set(165, Feature::nnp_assist);
        }

        // others

        enable_if_set(76, Feature::message_security_assist_extension3);
        enable_if_set(77, Feature::message_security_assist_extension4);
        enable_if_set(57, Feature::message_security_assist_extension5);
        enable_if_set(146, Feature::message_security_assist_extension8);
        enable_if_set(155, Feature::message_security_assist_extension9);
        enable_if_set(86, Feature::message_security_assist_extension12);

        enable_if_set(58, Feature::miscellaneous_extensions_2);
        enable_if_set(61, Feature::miscellaneous_extensions_3);
        enable_if_set(84, Feature::miscellaneous_extensions_4);

        enable_if_set(45, Feature::high_word);
        enable_if_set(73, Feature::transactional_execution);
        enable_if_set(133, Feature::guarded_storage);
        enable_if_set(150, Feature::enhanced_sort);
        enable_if_set(151, Feature::deflate_conversion);
        enable_if_set(201, Feature::concurrent_functions);
    }

    value
}
