use super::HidParseError;

// ============================================================================
// Wire Format: Raw Input Envelope (Drivers → Bristle)
// ============================================================================

/// Input device kind
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDeviceKind {
    Keyboard = 1,
    Mouse = 2,
    Consumer = 3,
    Gamepad = 4,
}

impl InputDeviceKind {
    pub fn from_raw(value: u8) -> Result<Self, HidParseError> {
        match value {
            1 => Ok(InputDeviceKind::Keyboard),
            2 => Ok(InputDeviceKind::Mouse),
            3 => Ok(InputDeviceKind::Consumer),
            4 => Ok(InputDeviceKind::Gamepad),
            _ => Err(HidParseError::BadKind),
        }
    }
}

/// Raw input envelope from drivers (16 bytes header + payload)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct RawInputEnvelope {
    pub device_id: u64,    // ThingId of device
    pub timestamp_ns: u64, // Driver-side timestamp
    pub kind: u8,          // InputDeviceKind
    pub payload_len: u8,   // Length of payload
    pub _pad: [u8; 6],     // Alignment padding
                           // payload bytes follow
}

impl RawInputEnvelope {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.device_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[16] = self.kind;
        buf[17] = self.payload_len;
        buf[18..24].copy_from_slice(&self._pad);
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<(Self, &[u8]), HidParseError> {
        if bytes.len() < Self::SIZE {
            return Err(HidParseError::TooShort);
        }
        let envelope = Self {
            device_id: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            timestamp_ns: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            kind: bytes[16],
            payload_len: bytes[17],
            _pad: bytes[18..24].try_into().unwrap(),
        };

        InputDeviceKind::from_raw(envelope.kind)?;
        let total_len = Self::SIZE + envelope.payload_len as usize;
        if bytes.len() != total_len {
            return Err(HidParseError::LengthMismatch);
        }
        Ok((envelope, &bytes[Self::SIZE..]))
    }
}

/// PS/2 keyboard payload (2 bytes)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Ps2KeyPayload {
    pub scancode: u8, // Raw scancode (bit 7 = break)
    pub flags: u8,    // bit0 = extended (E0 prefix)
}

impl Ps2KeyPayload {
    pub const SIZE: usize = core::mem::size_of::<Self>();

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        [self.scancode, self.flags]
    }

    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        Self {
            scancode: bytes[0],
            flags: bytes[1],
        }
    }
}
