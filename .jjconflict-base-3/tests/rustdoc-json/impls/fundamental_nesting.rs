// Companion to tests/rustdoc-html/impl/impl-fundamental-nesting.rs.
//
// Show traits implemented on fundamental types that wrap local ones: nested edition.

use std::pin::Pin;

pub struct Local;

// Nested fundamental + foreign Self.
/// from box local
impl From<Box<Local>> for String {
    fn from(_: Box<Local>) -> String {
        String::new()
    }
}
//@ set from_box_local = "$.index[?(@.docs=='from box local')].id"
//@ has "$.index[?(@.name=='Local')].inner.struct.impls[*]" $from_box_local

// Reference to a fundamental wrapper.
/// from ref box local
impl<'a> From<&'a Box<Local>> for u16 {
    fn from(_: &'a Box<Local>) -> u16 {
        0
    }
}
//@ set from_ref_box_local = "$.index[?(@.docs=='from ref box local')].id"
//@ has "$.index[?(@.name=='Local')].inner.struct.impls[*]" $from_ref_box_local

// Nested two levels deep in Self.
/// u32 for box box local
impl From<u32> for Box<Box<Local>> {
    fn from(_: u32) -> Box<Box<Local>> {
        Box::new(Box::new(Local))
    }
}
//@ set u32_for_box_box_local = "$.index[?(@.docs=='u32 for box box local')].id"
//@ has "$.index[?(@.name=='Local')].inner.struct.impls[*]" $u32_for_box_box_local

// Mixed fundamental wrappers in Self.
/// u64 for pin box local
impl From<u64> for Pin<Box<Local>> {
    fn from(_: u64) -> Pin<Box<Local>> {
        Pin::new(Box::new(Local))
    }
}
//@ set u64_for_pin_box_local = "$.index[?(@.docs=='u64 for pin box local')].id"
//@ has "$.index[?(@.name=='Local')].inner.struct.impls[*]" $u64_for_pin_box_local

// A non-fundamental wrapper must not associate the impl with Local, but the impl must still be
// listed on the trait itself.
pub trait Marker {}

/// marker for vec local
impl Marker for Vec<Local> {}
//@ set marker_for_vec_local = "$.index[?(@.docs=='marker for vec local')].id"
//@ !has "$.index[?(@.name=='Local')].inner.struct.impls[*]" $marker_for_vec_local
//@ has "$.index[?(@.name=='Marker')].inner.trait.implementations[*]" $marker_for_vec_local
