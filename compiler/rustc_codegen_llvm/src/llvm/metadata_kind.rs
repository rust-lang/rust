use libc::c_uint;

use crate::llvm::MetadataType;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(crate) struct MetadataKindId(c_uint);

impl From<MetadataType> for MetadataKindId {
    fn from(value: MetadataType) -> Self {
        Self(value as c_uint)
    }
}
