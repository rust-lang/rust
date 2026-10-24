// Followup to https://github.com/rust-lang/rust/issues/92940 and impl-box.rs.
//
// Show traits implemented on fundamental types that wrap local ones: nested edition.

#![crate_name = "foo"]

use std::pin::Pin;

pub struct Local;

//@ has 'foo/struct.Local.html'

// Nested fundamental + foreign Self.
//@ has '-' '//*[@id="impl-From%3CBox%3CLocal%3E%3E-for-String"]' 'impl From<Box<Local>> for String'
impl From<Box<Local>> for String {
    fn from(_: Box<Local>) -> String {
        String::new()
    }
}

// Also test with Pin.
//@ has '-' '//*[@id="impl-From%3CPin%3CLocal%3E%3E-for-u8"]' 'impl From<Pin<Local>> for u8'
impl From<Pin<Local>> for u8 {
    fn from(_: Pin<Local>) -> u8 {
        0
    }
}

// Reference to a fundamental wrapper.
//@ has '-' '//*[@id="impl-From%3C%26Box%3CLocal%3E%3E-for-u16"]' "impl<'a> From<&'a Box<Local>> for u16"
impl<'a> From<&'a Box<Local>> for u16 {
    fn from(_: &'a Box<Local>) -> u16 {
        0
    }
}

// Nested two levels deep in Self.
//@ has '-' '//*[@id="impl-From%3Cu32%3E-for-Box%3CBox%3CLocal%3E%3E"]' 'impl From<u32> for Box<Box<Local>>'
impl From<u32> for Box<Box<Local>> {
    fn from(_: u32) -> Box<Box<Local>> {
        Box::new(Box::new(Local))
    }
}

// Mixed fundamental wrappers in Self.
//@ has '-' '//*[@id="impl-From%3Cu64%3E-for-Pin%3CBox%3CLocal%3E%3E"]' 'impl From<u64> for Pin<Box<Local>>'
impl From<u64> for Pin<Box<Local>> {
    fn from(_: u64) -> Pin<Box<Local>> {
        Pin::new(Box::new(Local))
    }
}

// A non-fundamental wrapper must not show up on Local's page, but it should still be listed on the
// trait's own page.
pub trait Marker {}
//@ has 'foo/trait.Marker.html' '//*[@id="impl-Marker-for-Vec%3CLocal%3E"]' 'impl Marker for Vec<Local>'
//@ !has 'foo/struct.Local.html' '//*[@id="impl-Marker-for-Vec%3CLocal%3E"]' 'impl Marker for Vec<Local>'
impl Marker for Vec<Local> {}
