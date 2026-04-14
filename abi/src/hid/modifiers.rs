// ============================================================================
// Modifiers and Locks
// ============================================================================

/// Modifier key bitmask (currently held)
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods(pub u8);

impl Mods {
    pub const SHIFT: u8 = 1 << 0;
    pub const CTRL: u8 = 1 << 1;
    pub const ALT: u8 = 1 << 2;
    pub const META: u8 = 1 << 3;
    pub const ALTGR: u8 = 1 << 4;

    pub fn has_shift(self) -> bool {
        self.0 & Self::SHIFT != 0
    }
    pub fn has_ctrl(self) -> bool {
        self.0 & Self::CTRL != 0
    }
    pub fn has_alt(self) -> bool {
        self.0 & Self::ALT != 0
    }
    pub fn has_meta(self) -> bool {
        self.0 & Self::META != 0
    }
}

/// Lock state bitmask (toggle state)
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Locks(pub u8);

impl Locks {
    pub const CAPS: u8 = 1 << 0;
    pub const NUM: u8 = 1 << 1;
    pub const SCROLL: u8 = 1 << 2;
}
