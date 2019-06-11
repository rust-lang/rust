#![feature(nll)]
#![feature(rustc_private)]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

pub mod expand;

pub static ALLOCATOR_METHODS: &[AllocatorMethod] = &[
    AllocatorMethod {
        name: "alloc",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "dealloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout],
        output: AllocatorTy::Unit,
    },
    AllocatorMethod {
        name: "realloc",
        inputs: &[AllocatorTy::Ptr, AllocatorTy::Layout, AllocatorTy::Usize],
        output: AllocatorTy::ResultPtr,
    },
    AllocatorMethod {
        name: "alloc_zeroed",
        inputs: &[AllocatorTy::Layout],
        output: AllocatorTy::ResultPtr,
    },
];

pub struct AllocatorMethod {
    pub name: &'static str,
    pub inputs: &'static [AllocatorTy],
    pub output: AllocatorTy,
}

pub enum AllocatorTy {
    Layout,
    Ptr,
    ResultPtr,
    Unit,
    Usize,
}
