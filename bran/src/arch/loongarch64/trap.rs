#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UserTrapFrame {
    // r1-r31 (r0 is zero)
    pub regs: [usize; 31],

    pub era: usize,   // PC
    pub prmd: usize,  // Status
    pub badv: usize,  // Bad Vaddr
    pub estat: usize, // Exception Status
}
