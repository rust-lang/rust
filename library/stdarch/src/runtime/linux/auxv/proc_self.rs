//! Reads the ELF Auxiliary Vector from `/proc/self/auxv`.

use coresimd::__vendor_runtime::__runtime::linux::auxv;
use self::auxv::{AuxVec, AT_HWCAP};

use std::mem;

/// Tries to read the ELF Auxiliary Vector from `/proc/self/auxv`.
///
/// Errors if the file cannot be read. If a component of the auxvector
/// cannot be read, all the bits in its bitset are set to zero.
pub fn auxv() -> Result<AuxVec, ()> {
    auxv_from_file("/proc/self/auxv")
}

fn auxv_from_file(file: &str) -> Result<AuxVec, ()> {
    use std::io::Read;
    let mut file = ::std::fs::File::open(file).or_else(|_| Err(()))?;

    // See https://github.com/torvalds/linux/blob/v3.19/include/uapi/linux/auxvec.h
    //
    // The auxiliary vector contains at most 32 (key,value) fields: from
    // `AT_EXECFN = 31` to `AT_NULL = 0`. That is, a buffer of
    // 2*32 `usize` elements is enough to read the whole vector.
    let mut buf = [0usize; 64];
    {
        let raw: &mut [u8; 64 * mem::size_of::<usize>()] =
            unsafe { mem::transmute(&mut buf) };
        file.read(raw).or_else(|_| Err(()))?;
    }
    auxv_from_buf(&buf)
}

fn auxv_from_buf(buf: &[usize; 64]) -> Result<AuxVec, ()> {
    #[cfg(target_arch = "aarch64")]
    {
        for el in buf.chunks(2) {
            match el[0] {
                AT_HWCAP => return Ok(AuxVec { hwcap: el[1] }),
                _ => (),
            }
        }
    }

    #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
    {
        let mut hwcap = None;
        let mut hwcap2 = None;
        for el in buf.chunks(2) {
            match el[0] {
                AT_HWCAP => hwcap = Some(el[1]),
                auxv::AT_HWCAP2 => hwcap2 = Some(el[1]),
                _ => (),
            }
        }
        if hwcap.is_some() && hwcap2.is_some() {
            return Ok(AuxVec {
                hwcap: hwcap.unwrap(),
                hwcap2: hwcap2.unwrap(),
            });
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    extern crate auxv as auxv_crate;
    use super::*;

    // Reads the Auxiliary Vector key from /proc/self/auxv
    // using the auxv crate.
    fn auxv_crate_get(key: usize) -> Option<usize> {
        use self::auxv_crate::AuxvType;
        use self::auxv_crate::procfs::search_procfs_auxv;
        let k = key as AuxvType;
        match search_procfs_auxv(&[k]) {
            Ok(v) => Some(v[&k] as usize),
            Err(_) => None,
        }
    }

    #[test]
    fn auxv_dump() {
        if let Ok(auxvec) = auxv() {
            println!("{:?}", auxvec);
        } else {
            println!("reading /proc/self/auxv failed!");
        }
    }

    #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
    #[test]
    fn auxv_crate() {
        let v = auxv();
        if let Some(hwcap) = auxv_crate_get(AT_HWCAP) {
            assert_eq!(v.unwrap().hwcap, hwcap);
        }
        if let Some(hwcap2) = auxv_crate_get(auxv::AT_HWCAP2) {
            assert_eq!(v.unwrap().hwcap2, hwcap2);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn auxv_crate() {
        let v = auxv();
        if let Some(hwcap) = auxv_crate_get(AT_HWCAP) {
            assert_eq!(v.unwrap().hwcap, hwcap);
        }
    }

    #[cfg(all(target_arch = "arm", target_pointer_width = "32"))]
    #[test]
    fn linux_rpi3() {
        let v = auxv_from_file("src/runtime/linux/test_data/linux-rpi3.auxv")
            .unwrap();
        assert_eq!(v.hwcap, 4174038);
        assert_eq!(v.hwcap2, 16);
    }

    #[cfg(all(target_arch = "arm", target_pointer_width = "32"))]
    #[test]
    #[should_panic]
    fn linux_macos_vb() {
        let _ = auxv_from_file(
                "src/runtime/linux/test_data/macos-virtualbox-linux-x86-4850HQ.auxv"
            ).unwrap();
        // this file is incomplete (contains hwcap but not hwcap2), we
        // want to fall back to /proc/cpuinfo in this case, so
        // reading should fail. assert_eq!(v.hwcap, 126614527);
        // assert_eq!(v.hwcap2, 0);
    }

    #[cfg(all(target_arch = "aarch64", target_pointer_width = "64"))]
    #[test]
    fn linux_x64() {
        let v = auxv_from_file(
            "src/runtime/linux/test_data/linux-x64-i7-6850k.auxv",
        ).unwrap();
        assert_eq!(v.hwcap, 3219913727);
    }
}
