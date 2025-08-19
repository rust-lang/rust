use super::*;

#[cfg(feature = "std_detect_file_io")]
mod auxv_from_file {
    use super::auxvec::auxv_from_file;
    use super::*;
    // The baseline hwcaps used in the (artificial) auxv test files.
    fn baseline_hwcaps() -> AtHwcap {
        AtHwcap {
            fp: true,
            asimd: true,
            aes: true,
            pmull: true,
            sha1: true,
            sha2: true,
            crc32: true,
            atomics: true,
            fphp: true,
            asimdhp: true,
            asimdrdm: true,
            lrcpc: true,
            dcpop: true,
            asimddp: true,
            ssbs: true,
            ..AtHwcap::default()
        }
    }

    #[test]
    fn linux_empty_hwcap2_aarch64() {
        let file = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/detect/test_data/linux-empty-hwcap2-aarch64.auxv"
        );
        println!("file: {file}");
        let v = auxv_from_file(file).unwrap();
        println!("HWCAP : 0x{:0x}", v.hwcap);
        println!("HWCAP2: 0x{:0x}", v.hwcap2);
        assert_eq!(AtHwcap::from(v), baseline_hwcaps());
    }
    #[test]
    fn linux_no_hwcap2_aarch64() {
        let file = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/detect/test_data/linux-no-hwcap2-aarch64.auxv"
        );
        println!("file: {file}");
        let v = auxv_from_file(file).unwrap();
        println!("HWCAP : 0x{:0x}", v.hwcap);
        println!("HWCAP2: 0x{:0x}", v.hwcap2);
        assert_eq!(AtHwcap::from(v), baseline_hwcaps());
    }
    #[test]
    fn linux_hwcap2_aarch64() {
        let file =
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-hwcap2-aarch64.auxv");
        println!("file: {file}");
        let v = auxv_from_file(file).unwrap();
        println!("HWCAP : 0x{:0x}", v.hwcap);
        println!("HWCAP2: 0x{:0x}", v.hwcap2);
        assert_eq!(
            AtHwcap::from(v),
            AtHwcap {
                // Some other HWCAP bits.
                paca: true,
                pacg: true,
                // HWCAP2-only bits.
                dcpodp: true,
                frint: true,
                rng: true,
                bti: true,
                mte: true,
                ..baseline_hwcaps()
            }
        );
    }
}
