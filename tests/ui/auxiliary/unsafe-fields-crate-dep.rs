#![allow(incomplete_features)]
#![feature(unsafe_fields)]

pub struct WithUnsafeField {
    pub unsafe unsafe_field: u32,
    pub safe_field: u32,
}
