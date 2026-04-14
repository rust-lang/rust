// ============================================================================
// Event Types
// ============================================================================

/// Event type discriminant
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventType {
    KeyDown = 1,
    KeyUp = 2,
    PointerMove = 3,
    PointerButtonDown = 4,
    PointerButtonUp = 5,
    Scroll = 6,
    DeviceAdded = 7,
    DeviceRemoved = 8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HidParseError {
    TooShort,
    BadMagic,
    BadVersion,
    BadEventType,
    BadKind,
    LengthMismatch,
}

impl EventType {
    pub fn from_raw(value: u16) -> Result<Self, HidParseError> {
        match value {
            1 => Ok(EventType::KeyDown),
            2 => Ok(EventType::KeyUp),
            3 => Ok(EventType::PointerMove),
            4 => Ok(EventType::PointerButtonDown),
            5 => Ok(EventType::PointerButtonUp),
            6 => Ok(EventType::Scroll),
            7 => Ok(EventType::DeviceAdded),
            8 => Ok(EventType::DeviceRemoved),
            _ => Err(HidParseError::BadEventType),
        }
    }
}
