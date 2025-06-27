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

pub const ALLOC_ERROR_HANDLER: &str = "__rust_alloc_error_handler";
pub const ALLOC_ERROR_HANDLER_DEFAULT: &str = "__rdl_oom";
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
