//! Reads the ELF Auxiliary Vector using libc's `getauxval`.

use coresimd::__vendor_runtime::__runtime::linux::auxv;
use self::auxv::{AuxVec, AT_HWCAP};
use std::mem;

mod ffi {
    pub type F = unsafe extern "C" fn(usize) -> usize;
    #[allow(improper_ctypes)]
    extern "C" {
        #[linkage = "extern_weak"]
        pub static getauxval: *const ();
    }
}

/// Returns the value of the ELF Auxiliary Vector associated with `key`.
///
/// This only fails if the `getauxval` function is not linked.
///
/// The errno value is not checked, but if the key is not found
/// `getauxval` returns zero, and in that case, independently of
/// whether the key was found or no features were detected the
/// `auxv` function below returns `Err` to allow more
/// accurate run-time feature detection to run afterwards.
fn getauxval(key: usize) -> Result<usize, ()> {
    unsafe {
        if ffi::getauxval.is_null() {
            return Err(());
        }

        let ffi_getauxval: ffi::F = mem::transmute(ffi::getauxval);
        Ok(ffi_getauxval(key))
    }
}

/// Computes the entries of the Auxiliary Vector cache by
/// calling libc's `getauxval(3)`.
pub fn auxv() -> Result<AuxVec, ()> {
    if let Ok(hwcap) = getauxval(AT_HWCAP) {
        #[cfg(target_arch = "aarch64")]
        {
            if hwcap != 0 {
                return Ok(AuxVec { hwcap });
            }
        }
        #[cfg(any(target_arch = "arm", target_arch = "powerpc64"))]
        {
            if let Ok(hwcap2) = getauxval(auxv::AT_HWCAP2) {
                if hwcap != 0 && hwcap2 != 0 {
                    return Ok(AuxVec { hwcap, hwcap2 });
                }
            }
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    extern crate auxv as auxv_crate;
    use super::*;

    // Reads the Auxiliary Vector key from getauxval()
    // using the auxv crate.
    fn auxv_crate_get(key: usize) -> Option<usize> {
        use self::auxv_crate::AuxvType;
        use self::auxv_crate::getauxval::Getauxval;
        let q = auxv_crate::getauxval::NativeGetauxval {};
        match q.getauxval(key as AuxvType) {
            Ok(v) => Some(v as usize),
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
}
