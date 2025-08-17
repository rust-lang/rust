use super::*;

// FIXME: on mips/mips64 getauxval returns 0, and /proc/self/auxv
// does not always contain the AT_HWCAP key under qemu.
#[cfg(any(
    target_arch = "arm",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x",
))]
#[test]
fn auxv_crate() {
    let v = auxv();
    if let Ok(hwcap) = getauxval(AT_HWCAP) {
        let rt_hwcap = v.expect("failed to find hwcap key").hwcap;
        assert_eq!(rt_hwcap, hwcap);
    }

    // Targets with AT_HWCAP and AT_HWCAP2:
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ))]
    {
        if let Ok(hwcap2) = getauxval(AT_HWCAP2) {
            let rt_hwcap2 = v.expect("failed to find hwcap2 key").hwcap2;
            assert_eq!(rt_hwcap2, hwcap2);
        }
    }
}

#[test]
fn auxv_dump() {
    if let Ok(auxvec) = auxv() {
        println!("{:?}", auxvec);
    } else {
        println!("both getauxval() and reading /proc/self/auxv failed!");
    }
}

#[cfg(feature = "std_detect_file_io")]
cfg_select! {
    target_arch = "arm" => {
        // The tests below can be executed under qemu, where we do not have access to the test
        // files on disk, so we need to embed them with `include_bytes!`.
        #[test]
        fn linux_rpi3() {
            let auxv = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-rpi3.auxv"));
            let v = auxv_from_file_bytes(auxv).unwrap();
            assert_eq!(v.hwcap, 4174038);
            assert_eq!(v.hwcap2, 16);
        }

        #[test]
        fn linux_macos_vb() {
            let auxv = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/macos-virtualbox-linux-x86-4850HQ.auxv"));
            // The file contains HWCAP but not HWCAP2. In that case, we treat HWCAP2 as zero.
            let v = auxv_from_file_bytes(auxv).unwrap();
            assert_eq!(v.hwcap, 126614527);
            assert_eq!(v.hwcap2, 0);
        }
    }
    target_arch = "aarch64" => {
        #[cfg(target_endian = "little")]
        #[test]
        fn linux_artificial_aarch64() {
            let auxv = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-artificial-aarch64.auxv"));
            let v = auxv_from_file_bytes(auxv).unwrap();
            assert_eq!(v.hwcap, 0x0123456789abcdef);
            assert_eq!(v.hwcap2, 0x02468ace13579bdf);
        }
        #[cfg(target_endian = "little")]
        #[test]
        fn linux_no_hwcap2_aarch64() {
            let auxv = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/detect/test_data/linux-no-hwcap2-aarch64.auxv"));
            let v = auxv_from_file_bytes(auxv).unwrap();
            // An absent HWCAP2 is treated as zero, and does not prevent acceptance of HWCAP.
            assert_ne!(v.hwcap, 0);
            assert_eq!(v.hwcap2, 0);
        }
    }
    _ => {}
}

#[test]
#[cfg(feature = "std_detect_file_io")]
fn auxv_dump_procfs() {
    if let Ok(auxvec) = auxv_from_file("/proc/self/auxv") {
        println!("{:?}", auxvec);
    } else {
        println!("reading /proc/self/auxv failed!");
    }
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "arm",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "s390x",
))]
#[test]
#[cfg(feature = "std_detect_file_io")]
fn auxv_crate_procfs() {
    if let Ok(procfs_auxv) = auxv_from_file("/proc/self/auxv") {
        assert_eq!(auxv().unwrap(), procfs_auxv);
    }
}
