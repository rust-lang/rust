#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
pub struct UserTrapFrame {
    // General purpose registers pushed by `pushall` or similar
    // ABI order usually depends on push implementation.
    // Let's assume standard push order (last pushed is top of stack/first in struct).
    // If we push: rax, rbx, rcx, rdx, rsi, rdi, rbp, r8-r15
    // Then struct order is r15 first.
    pub r15: usize,
    pub r14: usize,
    pub r13: usize,
    pub r12: usize,
    pub r11: usize,
    pub r10: usize,
    pub r9: usize,
    pub r8: usize,
    pub rbp: usize,
    pub rdi: usize, // System V: RDI is arg0
    pub rsi: usize,
    pub rdx: usize,
    pub rcx: usize,
    pub rbx: usize,
    pub rax: usize,

    // Exception info
    pub error_code: usize,
    pub int_no: usize, // Often useful to distinguish trap type

    // Hardware frame via IRET
    pub rip: usize,
    pub cs: usize,
    pub rflags: usize,
    pub rsp: usize,
    pub ss: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::x86_64::gdt::{USER_CODE_SEL, USER_DATA_SEL};

    #[test]
    fn test_user_trap_frame_selectors() {
        let frame = UserTrapFrame {
            cs: USER_CODE_SEL as usize,
            ss: USER_DATA_SEL as usize,
            ..Default::default()
        };

        assert_eq!(frame.cs, 0x2B, "User CS should be 0x2B");
        assert_eq!(frame.ss, 0x23, "User SS should be 0x23");

        // Assert RPL=3 (bits 0 and 1)
        assert_eq!(frame.cs & 3, 3, "User CS must have RPL=3");
        assert_eq!(frame.ss & 3, 3, "User SS must have RPL=3");
    }
}
