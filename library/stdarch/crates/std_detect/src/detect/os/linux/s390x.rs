//! Run-time feature detection for s390x on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector
pub(crate) fn detect_features() -> cache::Initializer {
    if let Ok(auxv) = auxvec::auxv() {
        let hwcap: AtHwcap = auxv.into();
        return hwcap.cache();
    }

    cache::Initializer::default()
}

/// These values are part of the platform-specific [asm/elf.h][kernel], and are a selection of the
/// fields found in the [Facility Indications].
///
/// [Facility Indications]: https://www.ibm.com/support/pages/sites/default/files/2021-05/SA22-7871-10.pdf#page=63
/// [kernel]: https://github.com/torvalds/linux/blob/b62cef9a5c673f1b8083159f5dc03c1c5daced2f/arch/s390/include/asm/elf.h#L129
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

impl AtHwcap {
    /// Initializes the cache from the feature bits.
    fn cache(self) -> cache::Initializer {
        let mut value = cache::Initializer::default();
        {
            let mut enable_feature = |f, enable| {
                if enable {
                    value.set(f as u32);
                }
            };

            // vector and related

            // bit 129 of the extended facility list
            enable_feature(Feature::vector, self.vxrs);

            // bit 135 of the extended facility list
            enable_feature(Feature::vector_enhancements_1, self.vxrs_ext);

            // bit 148 of the extended facility list
            enable_feature(Feature::vector_enhancements_2, self.vxrs_ext2);

            // bit 134 of the extended facility list
            enable_feature(Feature::vector_packed_decimal, self.vxrs_bcd);

            // bit 152 of the extended facility list
            enable_feature(Feature::vector_packed_decimal_enhancement, self.vxrs_pde);

            // bit 192 of the extended facility list
            enable_feature(Feature::vector_packed_decimal_enhancement_2, self.vxrs_pde2);

            // bit 165 of the extended facility list
            enable_feature(Feature::nnp_assist, self.nnpa);

            // others

            // bit 45 of the extended facility list
            enable_feature(Feature::high_word, self.high_gprs);

            // bit 73 of the extended facility list
            enable_feature(Feature::transactional_execution, self.te);

            // bit 133 of the extended facility list
            enable_feature(Feature::guarded_storage, self.gs);

            // bit 150 of the extended facility list
            enable_feature(Feature::enhanced_sort, self.sort);

            // bit 151 of the extended facility list
            enable_feature(Feature::deflate_conversion, self.dflt);
        }
        value
    }
}
