//! PS/2 Scancode to HID Key Normalization
//!
//! Converts PS/2 Set 1 scancodes to normalized Key values.
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::hid::Key;

/// Convert PS/2 Set 1 scancode to normalized Key
///
/// # Arguments
/// * `scancode` - Raw scancode (7 bits, without break bit)
/// * `extended` - True if preceded by E0 prefix
pub fn ps2_to_key(scancode: u8, extended: bool) -> Key {
    if extended {
        // Extended scancodes (after E0 prefix)
        match scancode {
            0x48 => Key::Up,
            0x50 => Key::Down,
            0x4B => Key::Left,
            0x4D => Key::Right,
            0x47 => Key::Home,
            0x4F => Key::End,
            0x49 => Key::PageUp,
            0x51 => Key::PageDown,
            0x52 => Key::Insert,
            0x53 => Key::Delete,
            0x1D => Key::RightCtrl,
            0x38 => Key::RightAlt,
            _ => Key::Unknown,
        }
    } else {
        // Standard Set 1 scancodes
        match scancode {
            // Row 1: Esc, F1-F12
            0x01 => Key::Escape,
            0x3B => Key::F1,
            0x3C => Key::F2,
            0x3D => Key::F3,
            0x3E => Key::F4,
            0x3F => Key::F5,
            0x40 => Key::F6,
            0x41 => Key::F7,
            0x42 => Key::F8,
            0x43 => Key::F9,
            0x44 => Key::F10,
            0x57 => Key::F11,
            0x58 => Key::F12,

            // Row 2: ` 1-0 - = Bksp
            0x29 => Key::Grave,
            0x02 => Key::Num1,
            0x03 => Key::Num2,
            0x04 => Key::Num3,
            0x05 => Key::Num4,
            0x06 => Key::Num5,
            0x07 => Key::Num6,
            0x08 => Key::Num7,
            0x09 => Key::Num8,
            0x0A => Key::Num9,
            0x0B => Key::Num0,
            0x0C => Key::Minus,
            0x0D => Key::Equal,
            0x0E => Key::Backspace,

            // Row 3: Tab Q-P [ ] \
            0x0F => Key::Tab,
            0x10 => Key::Q,
            0x11 => Key::W,
            0x12 => Key::E,
            0x13 => Key::R,
            0x14 => Key::T,
            0x15 => Key::Y,
            0x16 => Key::U,
            0x17 => Key::I,
            0x18 => Key::O,
            0x19 => Key::P,
            0x1A => Key::LeftBracket,
            0x1B => Key::RightBracket,
            0x2B => Key::Backslash,

            // Row 4: Caps A-L ; ' Enter
            0x3A => Key::CapsLock,
            0x1E => Key::A,
            0x1F => Key::S,
            0x20 => Key::D,
            0x21 => Key::F,
            0x22 => Key::G,
            0x23 => Key::H,
            0x24 => Key::J,
            0x25 => Key::K,
            0x26 => Key::L,
            0x27 => Key::Semicolon,
            0x28 => Key::Quote,
            0x1C => Key::Enter,

            // Row 5: LShift Z-M , . / RShift
            0x2A => Key::LeftShift,
            0x2C => Key::Z,
            0x2D => Key::X,
            0x2E => Key::C,
            0x2F => Key::V,
            0x30 => Key::B,
            0x31 => Key::N,
            0x32 => Key::M,
            0x33 => Key::Comma,
            0x34 => Key::Period,
            0x35 => Key::Slash,
            0x36 => Key::RightShift,

            // Row 6: LCtrl, LAlt, Space
            0x1D => Key::LeftCtrl,
            0x38 => Key::LeftAlt,
            0x39 => Key::Space,

            _ => Key::Unknown,
        }
    }
}
