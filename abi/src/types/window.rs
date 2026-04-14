use crate::{BlobId, SymbolId, ThingId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct Window {
    pub title: BlobId,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}
