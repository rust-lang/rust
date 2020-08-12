use rustc_span::symbol::{sym, Symbol};

#[derive(Clone, Copy)]
pub enum AllocatorKind {
    Global,
    Default,
}

impl AllocatorKind {
    pub fn fn_name(&self, base: Symbol) -> String {
        match *self {
            AllocatorKind::Global => format!("__rg_{}", base),
            AllocatorKind::Default => format!("__rdl_{}", base),
        }
    }
}

pub enum AllocatorTy {
    Layout,
    Ptr,
    ResultPtr,
    Unit,
    Usize,
}

pub struct AllocatorMethod {
    pub name: Symbol,
    pub inputs: &'static [AllocatorTy],
    pub output: AllocatorTy,
}

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: sym::alloc,
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::dealloc,
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: sym::realloc,
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Usize],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::alloc_zeroed,
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
];
