use crate::{BlobId, SymbolId, ThingId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct Font {
    pub id: ThingId,
    pub family: SymbolId,
    pub style: SymbolId,
    pub blob: BlobId,
}
