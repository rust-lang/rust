#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UserTrapFrame {
    // x1-x31 (x0 is zero, not saved)
    pub regs: [usize; 31],

    pub sstatus: usize,
    pub sepc: usize,
    pub stval: usize,
    pub scause: usize,
}
