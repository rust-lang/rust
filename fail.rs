#![no_core]
#![feature(no_core)]
#![allow(internal_features)]
#![feature(lang_items)]

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

mod core_simd {
    mod vector {
        pub struct Simd {}
    }
    pub mod simd {
        pub use crate::core_simd::vector::*;
    }
}

pub mod simd {
    pub use crate::core_simd::simd::*;
}

mod fail {
    use crate::simd::Simd;
}
