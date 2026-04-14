use crate::{BlobId, SymbolId, ThingId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct Task {
    pub id: ThingId,
    pub name: SymbolId,
    pub parent: ThingId,
    pub state: u32,
}
