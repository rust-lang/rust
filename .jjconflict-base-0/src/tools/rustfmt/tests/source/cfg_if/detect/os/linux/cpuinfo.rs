//! Parses /proc/cpuinfo
#![cfg_attr(not(target_arch = "arm"), allow(dead_code))]

extern crate std;
use self::std::{prelude::v1::*, fs::File, io, io::Read};

/// cpuinfo
pub(crate) struct CpuInfo {
    raw: String,
}

impl CpuInfo {
    /// Reads /proc/cpuinfo into CpuInfo.
    pub(crate) fn new() -> Result<Self, io::Error> {
        let mut file = File::open("/proc/cpuinfo")?;
        let mut cpui = Self { raw: String::new() };
        file.read_to_string(&mut cpui.raw)?;
        Ok(cpui)
    }
    /// Returns the value of the cpuinfo `field`.
    pub(crate) fn field(&self, field: &str) -> CpuInfoField {
        for l in self.raw.lines() {
            if l.trim().starts_with(field) {
                return CpuInfoField::new(l.split(": ").nth(1));
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
    fn from_str(other: &str) -> Result<Self, ::std::io::Error> {
        Ok(Self {
            raw: String::from(other),
        })
    }
}

/// Field of cpuinfo
#[derive(Debug)]
pub(crate) struct CpuInfoField<'a>(Option<&'a str>);

impl<'a> PartialEq<&'a str> for CpuInfoField<'a> {
    fn eq(&self, other: &&'a str) -> bool {
        match self.0 {
            None => other.is_empty(),
            Some(f) => f == other.trim(),
        }
    }
}

impl<'a> CpuInfoField<'a> {
    pub(crate) fn new<'b>(v: Option<&'b str>) -> CpuInfoField<'b> {
        match v {
            None => CpuInfoField::<'b>(None),
            Some(f) => CpuInfoField::<'b>(Some(f.trim())),
        }
    }
    /// Does the field exist?
    #[cfg(test)]
    pub(crate) fn exists(&self) -> bool {
        self.0.is_some()
    }
    /// Does the field contain `other`?
    pub(crate) fn has(&self, other: &str) -> bool {
        match self.0 {
            None => other.is_empty(),
            Some(f) => {
                let other = other.trim();
                for v in f.split(' ') {
                    if v == other {
                        return true;
                    }
                }
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
