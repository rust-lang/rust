use rustc_macros::StableHash;
use rustc_span::{Symbol, sym};

#[derive(Clone, Debug, Copy, Eq, PartialEq, StableHash)]
pub enum AllocatorKind {
    /// Use `#[global_allocator]` as global allocator.
    Global,
    /// Use the default implementation in libstd as global allocator.
    Default,
}

pub fn global_fn_name(base: Symbol) -> String {
    format!("__rust_{base}")
}

pub fn default_fn_name(base: Symbol) -> String {
    format!("__rdl_{base}")
}

/// Returns the internal name of the default implementation of the token-enabled version of an
/// allocator method (i.e., the counterpart of `default_fn_name` for the methods in
/// `ALLOC_TOKEN_ALLOCATOR_METHODS`), for LLVM AllocToken and heap partitioning support.
pub fn default_fn_name_alloc_token(base: Symbol) -> String {
    format!("__rdl_{base}_with_token")
}

/// The prefix of the token-enabled versions of the allocation functions the `AllocTokenPass`
/// rewrites allocation calls to (i.e., the default value of its `-alloc-token-prefix` option),
/// for LLVM AllocToken and heap partitioning support.
pub const ALLOC_TOKEN_FN_PREFIX: &str = "__alloc_token_";

pub const ALLOC_ERROR_HANDLER: Symbol = sym::alloc_error_handler;
pub const NO_ALLOC_SHIM_IS_UNSTABLE: &str = "__rust_no_alloc_shim_is_unstable_v2";

/// Argument or return type for methods in the allocator shim
#[derive(Copy, Clone)]
pub enum AllocatorTy {
    Layout,
    Never,
    Ptr,
    ResultPtr,
    Unit,
    Usize,
}

/// Some allocator methods are known to the compiler: they act more like
/// intrinsics/language primitives than library-defined functions.
/// FIXME: ideally this would be derived from attributes like `#[rustc_allocator]`,
/// so we don't have two sources of truth.
#[derive(Copy, Clone, Debug)]
pub enum SpecialAllocatorMethod {
    Alloc,
    AllocZeroed,
    Dealloc,
    Realloc,
}

/// A method that will be codegened in the allocator shim.
#[derive(Copy, Clone)]
pub struct AllocatorMethod {
    pub name: Symbol,
    pub special: Option<SpecialAllocatorMethod>,
    pub inputs: &'static [AllocatorMethodInput],
    pub output: AllocatorTy,
    /// Whether this is the token-enabled version of the method (i.e., a method in
    /// `ALLOC_TOKEN_ALLOCATOR_METHODS`).
    pub with_token: bool,
    /// Whether the token-enabled method forwards to the non-token-enabled allocation function of
    /// the allocator registered with the `#[global_allocator]` attribute, ignoring the token
    /// identifier, instead of to the default implementation of the token-enabled version.
    pub forward_to_global: bool,
    /// The token identifier encoded in the allocation function name with the fast ABI (e.g.,
    /// `__alloc_token_0___rust_alloc`). `None` for the non-token-enabled methods and for the
    /// token-enabled methods without the fast ABI.
    pub token: Option<u32>,
}

pub struct AllocatorMethodInput {
    pub name: &'static str,
    pub ty: AllocatorTy,
}

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: sym::alloc,
        special: Some(SpecialAllocatorMethod::Alloc),
        inputs: &[AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout }],
        output: AllocatorTy::ResultPtr,
        with_token: false,
        forward_to_global: false,
        token: None,
    },
    AllocatorMethod {
        name: sym::dealloc,
        special: Some(SpecialAllocatorMethod::Dealloc),
        inputs: &[
            AllocatorMethodInput { name: "ptr", ty: AllocatorTy::Ptr },
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
        ],
        output: AllocatorTy::Unit,
        with_token: false,
        forward_to_global: false,
        token: None,
    },
    AllocatorMethod {
        name: sym::realloc,
        special: Some(SpecialAllocatorMethod::Realloc),
        inputs: &[
            AllocatorMethodInput { name: "ptr", ty: AllocatorTy::Ptr },
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
            AllocatorMethodInput { name: "new_size", ty: AllocatorTy::Usize },
        ],
        output: AllocatorTy::ResultPtr,
        with_token: false,
        forward_to_global: false,
        token: None,
    },
    AllocatorMethod {
        name: sym::alloc_zeroed,
        special: Some(SpecialAllocatorMethod::AllocZeroed),
        inputs: &[AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout }],
        output: AllocatorTy::ResultPtr,
        with_token: false,
        forward_to_global: false,
        token: None,
    },
];

/// Token-enabled versions of the allocator methods for LLVM AllocToken and heap partitioning
/// support. Unlike `ALLOCATOR_METHODS`, `dealloc` has no token-enabled version, because
/// deallocation functions are not rewritten (i.e., the allocator determines the partition of a
/// given pointer at deallocation time from the pointer itself).
pub static ALLOC_TOKEN_ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: sym::alloc,
        special: Some(SpecialAllocatorMethod::Alloc),
        inputs: &[
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
            AllocatorMethodInput { name: "token", ty: AllocatorTy::Usize },
        ],
        output: AllocatorTy::ResultPtr,
        with_token: true,
        forward_to_global: false,
        token: None,
    },
    AllocatorMethod {
        name: sym::realloc,
        special: Some(SpecialAllocatorMethod::Realloc),
        inputs: &[
            AllocatorMethodInput { name: "ptr", ty: AllocatorTy::Ptr },
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
            AllocatorMethodInput { name: "new_size", ty: AllocatorTy::Usize },
            AllocatorMethodInput { name: "token", ty: AllocatorTy::Usize },
        ],
        output: AllocatorTy::ResultPtr,
        with_token: true,
        forward_to_global: false,
        token: None,
    },
    AllocatorMethod {
        name: sym::alloc_zeroed,
        special: Some(SpecialAllocatorMethod::AllocZeroed),
        inputs: &[
            AllocatorMethodInput { name: "layout", ty: AllocatorTy::Layout },
            AllocatorMethodInput { name: "token", ty: AllocatorTy::Usize },
        ],
        output: AllocatorTy::ResultPtr,
        with_token: true,
        forward_to_global: false,
        token: None,
    },
];
