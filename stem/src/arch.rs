#[cfg(target_arch = "x86_64")]
use core::arch::asm;

#[derive(Debug, Clone, Copy)]
pub struct WhoAmI {
    pub cs: u16,
    pub ss: u16,
    pub cpl: u8,
    pub rsp: u64,
    pub rip: u64,
    pub rflags: u64,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn whoami() -> WhoAmI {
    let cs: u16;
    let ss: u16;
    let rsp: u64;
    let rip: u64;
    let rflags: u64;

    unsafe {
        asm!(
            "mov {cs_out:x}, cs",
            "mov {ss_out:x}, ss",
            "mov {rsp_out}, rsp",
            "lea {rip_out}, [rip]",
            "pushfq",
            "pop {rflags_out}",
            cs_out = out(reg) cs,
            ss_out = out(reg) ss,
            rsp_out = out(reg) rsp,
            rip_out = out(reg) rip,
            rflags_out = out(reg) rflags,
        );
    }

    WhoAmI {
        cs,
        ss,
        cpl: (cs & 0x3) as u8,
        rsp,
        rip,
        rflags,
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn whoami() -> WhoAmI {
    WhoAmI {
        cs: 0,
        ss: 0,
        cpl: 0,
        rsp: 0,
        rip: 0,
        rflags: 0,
    }
}
