use rustc_macros::HashStable_Generic;
use rustc_span::{Symbol, sym};

#[derive(Clone, Debug, Copy, Eq, PartialEq, HashStable_Generic)]
pub enum AllocatorKind {
    Global,
    Default,
}

pub fn global_fn_name(base: Symbol) -> String {
    format!("__rust_{base}")
}

pub fn default_fn_name(base: Symbol) -> String {
    format!("__rdl_{base}")
}

pub fn alloc_error_handler_name(alloc_error_handler_kind: AllocatorKind) -> &'static str {
    match alloc_error_handler_kind {
        AllocatorKind::Global => "__rg_oom",
        AllocatorKind::Default => "__rdl_oom",
    }
}

pub const NO_ALLOC_SHIM_IS_UNSTABLE: &str = "__rust_no_alloc_shim_is_unstable_v2";

pub enum AllocatorTy {
    Layout,
    Ptr,
    ResultPtr,
    Unit,
    Usize,
}

pub struct AllocatorMethod {
    pub name: Symbol,
    pub inputs: &'static [AllocatorMethodInput],
    pub output: AllocatorTy,
}

pub struct AllocatorMethodInput {
    pub name: &'static str,
    pub ty: AllocatorTy,
}

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: sym::alloc,
        inputs: &[AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout }],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::dealloc,
        inputs: &[
            AllocatorMethodInput { name: "ptr", ty: AllocatorTy::Ptr },
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
        ],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: sym::realloc,
        inputs: &[
            AllocatorMethodInput { name: "ptr", ty: AllocatorTy::Ptr },
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
            AllocatorMethodInput { name: "new_size", ty: AllocatorTy::Usize },
        ],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: sym::alloc_zeroed,
        inputs: &[AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout }],
        output: AllocatorTy::ResultPtr,
    },
];
