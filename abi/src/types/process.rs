use crate::{BlobId, SymbolId, ThingId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct Process {
    pub id: ThingId,
    pub name: SymbolId,
    pub pid: u64,
}
