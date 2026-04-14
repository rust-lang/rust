//! Thigmonasty: Keyboard State Engine
//!
//! The "reflex arc" for keyboard input:
//! - Tracks pressed keys
//! - Maintains modifier state
//! - Generates repeat events
//! - Emits KeyDown/KeyUp edges
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use crate::normalizer::ps2_to_key;
use abi::hid::{Key, Mods};

/// Keyboard state tracker
pub struct KeyboardState {
    /// Current modifier bitmask
    mods: u8,
    /// E0 extended prefix pending
    e0_prefix: bool,
    /// Currently pressed keys (simple bitset for common keys)
    pressed: [u64; 4], // 256 bits
}

/// Edge event emitted by the keyboard state machine
#[derive(Clone, Copy, Debug)]
pub enum KeyEdge {
    Down { key: Key, mods: Mods, repeat: bool },
    Up { key: Key, mods: Mods },
}

impl KeyboardState {
    pub fn new() -> Self {
        Self {
            mods: 0,
            e0_prefix: false,
            pressed: [0; 4],
        }
    }

    /// Process a raw PS/2 scancode byte, returning edge event if any
    pub fn process_ps2(&mut self, byte: u8) -> Option<KeyEdge> {
        // Handle E0 extended prefix
        if byte == 0xE0 {
            self.e0_prefix = true;
            return None;
        }

        let is_break = byte & 0x80 != 0;
        let scancode = byte & 0x7F;
        let extended = self.e0_prefix;
        self.e0_prefix = false;

        // Convert to normalized key
        let key = ps2_to_key(scancode, extended);

        // Update tracking bitset and determine if this is a repeat or new edge
        let key_idx = key as u16 as usize;
        let (word_idx, bit_mask) = (key_idx / 64, 1 << (key_idx % 64));

        let mut was_pressed = false;
        if word_idx < 4 {
            was_pressed = (self.pressed[word_idx] & bit_mask) != 0;
            if is_break {
                self.pressed[word_idx] &= !bit_mask;
            } else {
                self.pressed[word_idx] |= bit_mask;
            }
        }

        // Update modifier state (now with correct bitset)
        self.update_mods(key, !is_break);

        // Emit edge event
        if is_break {
            Some(KeyEdge::Up {
                key,
                mods: Mods(self.mods),
            })
        } else {
            Some(KeyEdge::Down {
                key,
                mods: Mods(self.mods),
                repeat: was_pressed,
            })
        }
    }

    fn update_mods(&mut self, key: Key, pressed: bool) {
        match key {
            Key::LeftShift | Key::RightShift => {
                if pressed {
                    self.mods |= Mods::SHIFT;
                } else {
                    // Only clear if BOTH Shift keys are now released
                    if !self.is_key_pressed(Key::LeftShift) && !self.is_key_pressed(Key::RightShift)
                    {
                        self.mods &= !Mods::SHIFT;
                    }
                }
            }
            Key::LeftCtrl | Key::RightCtrl => {
                if pressed {
                    self.mods |= Mods::CTRL;
                } else {
                    if !self.is_key_pressed(Key::LeftCtrl) && !self.is_key_pressed(Key::RightCtrl) {
                        self.mods &= !Mods::CTRL;
                    }
                }
            }
            Key::LeftAlt | Key::RightAlt => {
                // Note: RightAlt is handled as ALTGR in some layouts, but for OS-level
                // modifiers we often want both to act as ALT.
                // Bristle currently maps RightAlt to ALTGR bit.
                if key == Key::LeftAlt {
                    if pressed {
                        self.mods |= Mods::ALT;
                    } else {
                        self.mods &= !Mods::ALT;
                    }
                } else {
                    if pressed {
                        self.mods |= Mods::ALTGR;
                    } else {
                        self.mods &= !Mods::ALTGR;
                    }
                }
            }
            Key::LeftMeta | Key::RightMeta => {
                if pressed {
                    self.mods |= Mods::META;
                } else {
                    if !self.is_key_pressed(Key::LeftMeta) && !self.is_key_pressed(Key::RightMeta) {
                        self.mods &= !Mods::META;
                    }
                }
            }
            _ => {}
        }
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        let key_idx = key as u16 as usize;
        let word_idx = key_idx / 64;
        let bit_idx = key_idx % 64;
        if word_idx < 4 {
            (self.pressed[word_idx] & (1 << bit_idx)) != 0
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub fn mods(&self) -> Mods {
        Mods(self.mods)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi::hid::{Key, Mods};

    #[test]
    fn test_shift_modifier() {
        let mut state = KeyboardState::new();

        // Press LeftShift (0x2A)
        let edge = state.process_ps2(0x2A).expect("Expected edge");
        match edge {
            KeyEdge::Down { key, mods, repeat } => {
                assert_eq!(key, Key::LeftShift);
                assert_eq!(mods.0, Mods::SHIFT);
                assert!(!repeat);
            }
            _ => panic!("Expected KeyDown"),
        }

        // Release LeftShift (0x2A | 0x80 = 0xAA)
        let edge = state.process_ps2(0xAA).expect("Expected edge");
        match edge {
            KeyEdge::Up { key, mods } => {
                assert_eq!(key, Key::LeftShift);
                assert_eq!(mods.0, 0); // Should be cleared now
            }
            _ => panic!("Expected KeyUp"),
        }
    }

    #[test]
    fn test_dual_shift() {
        let mut state = KeyboardState::new();

        // LShift down
        state.process_ps2(0x2A);
        assert_eq!(state.mods().0, Mods::SHIFT);

        // RShift down
        state.process_ps2(0x36);
        assert_eq!(state.mods().0, Mods::SHIFT);

        // LShift up
        state.process_ps2(0xAA);
        assert_eq!(state.mods().0, Mods::SHIFT); // Still shifted by RShift

        // RShift up
        state.process_ps2(0xB6);
        assert_eq!(state.mods().0, 0); // Finally clear
    }

    #[test]
    fn test_altgr() {
        let mut state = KeyboardState::new();

        // RAlt (E0 38)
        state.process_ps2(0xE0);
        state.process_ps2(0x38);
        assert_eq!(state.mods().0, Mods::ALTGR);

        // Release RAlt (E0 B8)
        state.process_ps2(0xE0);
        state.process_ps2(0xB8);
        assert_eq!(state.mods().0, 0);
    }
}
