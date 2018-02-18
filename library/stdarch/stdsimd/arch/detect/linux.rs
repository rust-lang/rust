
#![allow(dead_code)]

use core::mem;

use prelude::v1::*;
use fs::File;
use io::{self, Read};

/// Key to access the CPU Hardware capabilities bitfield.
pub const AT_HWCAP: usize = 16;
/// Key to access the CPU Hardware capabilities 2 bitfield.
pub const AT_HWCAP2: usize = 26;

/// Cache HWCAP bitfields of the ELF Auxiliary Vector.
///
/// If an entry cannot be read all the bits in the bitfield
/// are set to zero.
#[derive(Debug, Copy, Clone)]
pub struct AuxVec {
    pub hwcap: usize,
    #[cfg(not(target_arch = "aarch64"))]
    pub hwcap2: usize,
}

/// ELF Auxiliary Vector
///
/// The auxiliary vector is a memory region in a running ELF program's stack
/// composed of (key: usize, value: usize) pairs.
///
/// The keys used in the aux vector are platform dependent. For Linux, they are
/// defined in [linux/auxvec.h][auxvec_h]. The hardware capabilities of a given
/// CPU can be queried with the  `AT_HWCAP` and `AT_HWCAP2` keys.
///
/// There is no perfect way of reading the auxiliary vector.
///
/// - `coresimd`: if `getauxval` is available, `coresimd` will try to use it.
/// - `stdsimd`: if `getauxval` is not available, it will try to read
/// `/proc/self/auxv`, and if that fails it will try to read `/proc/cpuinfo`.
///
/// For more information about when `getauxval` is available check the great
/// [`auxv` crate documentation][auxv_docs].
///
/// [auxvec_h]: https://github.com/torvalds/linux/blob/master/include/uapi/linux/auxvec.h
/// [auxv_docs]: https://docs.rs/auxv/0.3.3/auxv/
pub fn auxv() -> Result<AuxVec, ()> {
    if !cfg!(target_os = "linux") {
        return Err(())
    }
    if let Ok(hwcap) = getauxval(AT_HWCAP) {
        #[cfg(target_arch = "aarch64")]
        {
            if hwcap != 0 {
                return Ok(AuxVec { hwcap });
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            if let Ok(hwcap2) = getauxval(AT_HWCAP2) {
                if hwcap != 0 && hwcap2 != 0 {
                    return Ok(AuxVec { hwcap, hwcap2 });
                }
            }
        }
    }

    return auxv_from_file("/proc/self/auxv");

    #[cfg(not(target_os = "linux"))]
    fn getauxval(_key: usize) -> Result<usize, ()> {
        Err(())
    }

    #[cfg(target_os = "linux")]
    fn getauxval(key: usize) -> Result<usize, ()> {
        use libc;

        pub type F = unsafe extern "C" fn(usize) -> usize;

        unsafe {
            let ptr = libc::dlsym(libc::RTLD_DEFAULT, "getauxval\0".as_ptr() as *const _);
            if ptr.is_null() {
                return Err(());
            }

            let ffi_getauxval: F = mem::transmute(ptr);
            Ok(ffi_getauxval(key))
        }
    }
}

fn auxv_from_file(file: &str) -> Result<AuxVec, ()> {
    let mut file = File::open(file).map_err(|_| ())?;

    // See https://github.com/torvalds/linux/blob/v3.19/include/uapi/linux/auxvec.h
    //
    // The auxiliary vector contains at most 32 (key,value) fields: from
    // `AT_EXECFN = 31` to `AT_NULL = 0`. That is, a buffer of
    // 2*32 `usize` elements is enough to read the whole vector.
    let mut buf = [0usize; 64];
    {
        let raw: &mut [u8; 64 * mem::size_of::<usize>()] =
            unsafe { mem::transmute(&mut buf) };
        file.read(raw).map_err(|_| ())?;
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

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut hwcap = None;
        let mut hwcap2 = None;
        for el in buf.chunks(2) {
            match el[0] {
                AT_HWCAP => hwcap = Some(el[1]),
                AT_HWCAP2 => hwcap2 = Some(el[1]),
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

    drop(buf);
    Err(())
}

/// cpuinfo
pub struct CpuInfo {
    raw: String,
}

impl CpuInfo {
    /// Reads /proc/cpuinfo into CpuInfo.
    pub fn new() -> Result<CpuInfo, io::Error> {
        let mut file = File::open("/proc/cpuinfo")?;
        let mut cpui = CpuInfo { raw: String::new() };
        file.read_to_string(&mut cpui.raw)?;
        Ok(cpui)
    }
    /// Returns the value of the cpuinfo `field`.
    pub fn field(&self, field: &str) -> CpuInfoField {
        for l in self.raw.lines() {
            if l.trim().starts_with(field) {
                return CpuInfoField::new(l.split(": ").skip(1).next());
            }
        }
        CpuInfoField(None)
    }

    /// Returns the `raw` contents of `/proc/cpuinfo`
    #[cfg(test)]
    fn raw(&self) -> &String {
        &self.raw
    }

    #[cfg(test)]
    fn from_str(other: &str) -> Result<CpuInfo, ::std::io::Error> {
        Ok(CpuInfo {
            raw: String::from(other),
        })
    }
}

/// Field of cpuinfo
#[derive(Debug)]
pub struct CpuInfoField<'a>(Option<&'a str>);

impl<'a> PartialEq<&'a str> for CpuInfoField<'a> {
    fn eq(&self, other: &&'a str) -> bool {
        match self.0 {
            None => other.len() == 0,
            Some(f) => f == other.trim(),
        }
    }
}

impl<'a> CpuInfoField<'a> {
    pub fn new<'b>(v: Option<&'b str>) -> CpuInfoField<'b> {
        match v {
            None => CpuInfoField::<'b>(None),
            Some(f) => CpuInfoField::<'b>(Some(f.trim())),
        }
    }
    /// Does the field exist?
    #[cfg(test)]
    pub fn exists(&self) -> bool {
        self.0.is_some()
    }
    /// Does the field contain `other`?
    pub fn has(&self, other: &str) -> bool {
        match self.0 {
            None => other.len() == 0,
            Some(f) => {
                let other = other.trim();
                for v in f.split(" ") {
                    if v == other {
                        return true;
                    }
                }
                false
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    extern crate auxv as auxv_crate;
    use super::*;

    // Reads the Auxiliary Vector key from /proc/self/auxv
    // using the auxv crate.
    fn auxv_crate_getprocfs(key: usize) -> Option<usize> {
        use self::auxv_crate::AuxvType;
        use self::auxv_crate::procfs::search_procfs_auxv;
        let k = key as AuxvType;
        match search_procfs_auxv(&[k]) {
            Ok(v) => Some(v[&k] as usize),
            Err(_) => None,
        }
    }

    // Reads the Auxiliary Vector key from getauxval()
    // using the auxv crate.
    fn auxv_crate_getauxval(key: usize) -> Option<usize> {
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

    #[test]
    fn auxv_crate() {
        if cfg!(target_arch = "x86") ||
            cfg!(target_arch = "x86_64") ||
            cfg!(target_arch = "powerpc") {
            return
        }
        let v = auxv();
        if let Some(hwcap) = auxv_crate_getauxval(AT_HWCAP) {
            assert_eq!(v.unwrap().hwcap, hwcap);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            if let Some(hwcap2) = auxv_crate_getauxval(AT_HWCAP2) {
                assert_eq!(v.unwrap().hwcap2, hwcap2);
            }
        }
    }

    #[cfg(target_arch = "arm")]
    #[test]
    fn linux_rpi3() {
        let v = auxv_from_file("../../stdsimd/arch/detect/test_data/linux-rpi3.auxv")
            .unwrap();
        assert_eq!(v.hwcap, 4174038);
        assert_eq!(v.hwcap2, 16);
    }

    #[cfg(target_arch = "arm")]
    #[test]
    #[should_panic]
    fn linux_macos_vb() {
        let _ = auxv_from_file(
                "../../stdsimd/arch/detect/test_data/macos-virtualbox-linux-x86-4850HQ.auxv"
            ).unwrap();
        // this file is incomplete (contains hwcap but not hwcap2), we
        // want to fall back to /proc/cpuinfo in this case, so
        // reading should fail. assert_eq!(v.hwcap, 126614527);
        // assert_eq!(v.hwcap2, 0);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn linux_x64() {
        let v = auxv_from_file(
            "../../stdsimd/arch/detect/test_data/linux-x64-i7-6850k.auxv",
        ).unwrap();
        assert_eq!(v.hwcap, 3219913727);
    }

    #[test]
    fn auxv_dump_procfs() {
        if let Ok(auxvec) = auxv_from_file("/proc/self/auxv") {
            println!("{:?}", auxvec);
        } else {
            println!("reading /proc/self/auxv failed!");
        }
    }

    #[test]
    fn auxv_crate_procfs() {
        if cfg!(target_arch = "x86") || cfg!(target_arch = "x86_64") {
            return
        }
        let v = auxv();
        if let Some(hwcap) = auxv_crate_getprocfs(AT_HWCAP) {
            assert_eq!(v.unwrap().hwcap, hwcap);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            if let Some(hwcap2) = auxv_crate_getprocfs(AT_HWCAP2) {
                assert_eq!(v.unwrap().hwcap2, hwcap2);
            }
        }
    }

    #[test]
    fn raw_dump() {
        let cpuinfo = CpuInfo::new().unwrap();
        if cpuinfo.field("vendor_id") == "GenuineIntel" {
            assert!(cpuinfo.field("flags").exists());
            assert!(!cpuinfo.field("vendor33_id").exists());
            assert!(cpuinfo.field("flags").has("sse"));
            assert!(!cpuinfo.field("flags").has("avx314"));
        }
        println!("{}", cpuinfo.raw());
    }

    const CORE_DUO_T6500: &str = r"processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 23
model name      : Intel(R) Core(TM)2 Duo CPU     T6500  @ 2.10GHz
stepping        : 10
microcode       : 0xa0b
cpu MHz         : 1600.000
cache size      : 2048 KB
physical id     : 0
siblings        : 2
core id         : 0
cpu cores       : 2
apicid          : 0
initial apicid  : 0
fdiv_bug        : no
hlt_bug         : no
f00f_bug        : no
coma_bug        : no
fpu             : yes
fpu_exception   : yes
cpuid level     : 13
wp              : yes
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx lm constant_tsc arch_perfmon pebs bts aperfmperf pni dtes64 monitor ds_cpl est tm2 ssse3 cx16 xtpr pdcm sse4_1 xsave lahf_lm dtherm
bogomips        : 4190.43
clflush size    : 64
cache_alignment : 64
address sizes   : 36 bits physical, 48 bits virtual
power management:
";

    #[test]
    fn core_duo_t6500() {
        let cpuinfo = CpuInfo::from_str(CORE_DUO_T6500).unwrap();
        assert_eq!(cpuinfo.field("vendor_id"), "GenuineIntel");
        assert_eq!(cpuinfo.field("cpu family"), "6");
        assert_eq!(cpuinfo.field("model"), "23");
        assert_eq!(
            cpuinfo.field("model name"),
            "Intel(R) Core(TM)2 Duo CPU     T6500  @ 2.10GHz"
        );
        assert_eq!(
            cpuinfo.field("flags"),
            "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe nx lm constant_tsc arch_perfmon pebs bts aperfmperf pni dtes64 monitor ds_cpl est tm2 ssse3 cx16 xtpr pdcm sse4_1 xsave lahf_lm dtherm"
        );
        assert!(cpuinfo.field("flags").has("fpu"));
        assert!(cpuinfo.field("flags").has("dtherm"));
        assert!(cpuinfo.field("flags").has("sse2"));
        assert!(!cpuinfo.field("flags").has("avx"));
    }

    const ARM_CORTEX_A53: &str =
        r"Processor   : AArch64 Processor rev 3 (aarch64)
        processor   : 0
        processor   : 1
        processor   : 2
        processor   : 3
        processor   : 4
        processor   : 5
        processor   : 6
        processor   : 7
        Features    : fp asimd evtstrm aes pmull sha1 sha2 crc32
        CPU implementer : 0x41
        CPU architecture: AArch64
        CPU variant : 0x0
        CPU part    : 0xd03
        CPU revision    : 3

        Hardware    : HiKey Development Board
        ";

    #[test]
    fn arm_cortex_a53() {
        let cpuinfo = CpuInfo::from_str(ARM_CORTEX_A53).unwrap();
        assert_eq!(
            cpuinfo.field("Processor"),
            "AArch64 Processor rev 3 (aarch64)"
        );
        assert_eq!(
            cpuinfo.field("Features"),
            "fp asimd evtstrm aes pmull sha1 sha2 crc32"
        );
        assert!(cpuinfo.field("Features").has("pmull"));
        assert!(!cpuinfo.field("Features").has("neon"));
        assert!(cpuinfo.field("Features").has("asimd"));
    }

    const ARM_CORTEX_A57: &str = r"Processor	: Cortex A57 Processor rev 1 (aarch64)
processor	: 0
processor	: 1
processor	: 2
processor	: 3
Features	: fp asimd aes pmull sha1 sha2 crc32 wp half thumb fastmult vfp edsp neon vfpv3 tlsi vfpv4 idiva idivt
CPU implementer	: 0x41
CPU architecture: 8
CPU variant	: 0x1
CPU part	: 0xd07
CPU revision	: 1";

    #[test]
    fn arm_cortex_a57() {
        let cpuinfo = CpuInfo::from_str(ARM_CORTEX_A57).unwrap();
        assert_eq!(
            cpuinfo.field("Processor"),
            "Cortex A57 Processor rev 1 (aarch64)"
        );
        assert_eq!(
            cpuinfo.field("Features"),
            "fp asimd aes pmull sha1 sha2 crc32 wp half thumb fastmult vfp edsp neon vfpv3 tlsi vfpv4 idiva idivt"
        );
        assert!(cpuinfo.field("Features").has("pmull"));
        assert!(cpuinfo.field("Features").has("neon"));
        assert!(cpuinfo.field("Features").has("asimd"));
    }

    const POWER8E_POWERKVM: &str = r"processor       : 0
cpu             : POWER8E (raw), altivec supported
clock           : 3425.000000MHz
revision        : 2.1 (pvr 004b 0201)

processor       : 1
cpu             : POWER8E (raw), altivec supported
clock           : 3425.000000MHz
revision        : 2.1 (pvr 004b 0201)

processor       : 2
cpu             : POWER8E (raw), altivec supported
clock           : 3425.000000MHz
revision        : 2.1 (pvr 004b 0201)

processor       : 3
cpu             : POWER8E (raw), altivec supported
clock           : 3425.000000MHz
revision        : 2.1 (pvr 004b 0201)

timebase        : 512000000
platform        : pSeries
model           : IBM pSeries (emulated by qemu)
machine         : CHRP IBM pSeries (emulated by qemu)";

    #[test]
    fn power8_powerkvm() {
        let cpuinfo = CpuInfo::from_str(POWER8E_POWERKVM).unwrap();
        assert_eq!(cpuinfo.field("cpu"), "POWER8E (raw), altivec supported");

        assert!(cpuinfo.field("cpu").has("altivec"));
    }

    const POWER5P: &str = r"processor       : 0
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 1
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 2
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 3
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 4
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 5
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 6
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

processor       : 7
cpu             : POWER5+ (gs)
clock           : 1900.098000MHz
revision        : 2.1 (pvr 003b 0201)

timebase        : 237331000
platform        : pSeries
machine         : CHRP IBM,9133-55A";

    #[test]
    fn power5p() {
        let cpuinfo = CpuInfo::from_str(POWER5P).unwrap();
        assert_eq!(cpuinfo.field("cpu"), "POWER5+ (gs)");

        assert!(!cpuinfo.field("cpu").has("altivec"));
    }
}
