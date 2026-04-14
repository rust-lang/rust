use super::{EventType, HidParseError, Key, Mods, BRISTLE_EVENT_MAGIC, BRISTLE_EVENT_VERSION};

// ============================================================================
// Wire Format: Bristle Event (Bristle → Apps)
// ============================================================================

/// Bristle event header (20 bytes)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BristleEventHeader {
    pub magic: u32,        // BRISTLE_EVENT_MAGIC
    pub version: u16,      // BRISTLE_EVENT_VERSION
    pub event_type: u16,   // EventType discriminant
    pub timestamp_ns: u64, // Monotonic timestamp
    pub payload_len: u32,  // Bytes following header
}

impl BristleEventHeader {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.event_type.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[16..20].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Result<Self, HidParseError> {
        let header = Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            version: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            event_type: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            timestamp_ns: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            payload_len: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
        };

        if header.magic != BRISTLE_EVENT_MAGIC {
            return Err(HidParseError::BadMagic);
        }
        if header.version != BRISTLE_EVENT_VERSION {
            return Err(HidParseError::BadVersion);
        }
        EventType::from_raw(header.event_type)?;
        Ok(header)
    }
}

/// KeyDown/KeyUp payload (4 bytes)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct KeyEventPayload {
    pub key: u16,  // Key enum value
    pub mods: u8,  // Mods bitmask
    pub flags: u8, // bit0 = repeat
}

impl KeyEventPayload {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn key(&self) -> Key {
        Key::from_raw(self.key)
    }
    pub fn mods(&self) -> Mods {
        Mods(self.mods)
    }
    pub fn is_repeat(&self) -> bool {
        self.flags & 1 != 0
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.key.to_le_bytes());
        buf[2] = self.mods;
        buf[3] = self.flags;
        buf
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            key: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            mods: bytes[2],
            flags: bytes[3],
        }
    }
}
