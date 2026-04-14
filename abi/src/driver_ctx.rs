use crate::ids::HandleId;
use crate::types::ThingId;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DriverCtx {
    pub device_id: ThingId,
    // Future: root_caps, etc.
}

/// Helper to convert ctx to raw arg (v0: fits in register)
impl DriverCtx {
    pub fn to_raw(self) -> usize {
        self.device_id.to_u64_lossy() as usize
    }

    pub fn from_raw(arg: usize) -> Self {
        Self {
            device_id: ThingId::from_u64(arg as u64),
        }
    }
}
