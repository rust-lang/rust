// Unknown attributes fall back to feature gated custom attributes.

#![feature(custom_inner_attributes)]

#![mutable_doc] //~ ERROR attribute `mutable_doc` is currently unknown

#[dance] mod a {} //~ ERROR attribute `dance` is currently unknown

#[dance] fn main() {} //~ ERROR attribute `dance` is currently unknown
