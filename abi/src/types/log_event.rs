use crate::{BlobId, SymbolId, ThingId};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C, packed)]
pub struct LogEvent {
    pub source: ThingId,
    pub timestamp_ns: u64,
    pub level: u32,
    pub message: BlobId,
}
