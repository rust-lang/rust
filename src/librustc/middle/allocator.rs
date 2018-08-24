#[derive(Clone, Copy)]
pub enum AllocatorKind {
    Global,
    DefaultLib,
    DefaultExe,
}

impl AllocatorKind {
    pub fn fn_name(&self, base: &str) -> String {
        match *self {
            AllocatorKind::Global => format!("__rg_{}", base),
            AllocatorKind::DefaultLib => format!("__rdl_{}", base),
            AllocatorKind::DefaultExe => format!("__rde_{}", base),
        }
    }
}
