//! Mouse Packet Processing
//!
//! Parses PS/2 mouse packets and emits pointer events.
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

/// Mouse button state
#[derive(Clone, Copy, Default)]
pub struct MouseState {
    left: bool,
    right: bool,
    middle: bool,
}

/// Pointer event emitted by mouse processing
#[derive(Clone, Copy, Debug)]
pub enum PointerEvent {
    Move { dx: i16, dy: i16 },
    ButtonDown { button: u8 },
    ButtonUp { button: u8 },
}

impl MouseState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a 3-byte PS/2 mouse packet, returning events
    /// Returns up to 4 events: move + up to 3 button changes
    pub fn process_packet(&mut self, packet: &[u8; 3]) -> ([Option<PointerEvent>; 4], usize) {
        let mut events = [None; 4];
        let mut count = 0;

        let flags = packet[0];
        let x_raw = packet[1];
        let y_raw = packet[2];

        // Check overflow - discard packet if overflowed
        if (flags & 0xC0) != 0 {
            return (events, 0);
        }

        // Compute signed deltas (9-bit signed values)
        let x_sign = (flags & 0x10) != 0;
        let y_sign = (flags & 0x20) != 0;

        let dx: i16 = if x_sign {
            (x_raw as i16) - 256
        } else {
            x_raw as i16
        };

        // Note: PS/2 Y axis is inverted (negative = up)
        let dy: i16 = if y_sign {
            -((y_raw as i16) - 256)
        } else {
            -(y_raw as i16)
        };

        // Emit move event if there's movement
        if dx != 0 || dy != 0 {
            events[count] = Some(PointerEvent::Move { dx, dy });
            count += 1;
        }

        // Button changes
        let new_left = (flags & 0x01) != 0;
        let new_right = (flags & 0x02) != 0;
        let new_middle = (flags & 0x04) != 0;

        if new_left != self.left {
            events[count] = Some(if new_left {
                PointerEvent::ButtonDown { button: 0 }
            } else {
                PointerEvent::ButtonUp { button: 0 }
            });
            count += 1;
            self.left = new_left;
        }

        if new_right != self.right {
            events[count] = Some(if new_right {
                PointerEvent::ButtonDown { button: 1 }
            } else {
                PointerEvent::ButtonUp { button: 1 }
            });
            count += 1;
            self.right = new_right;
        }

        if new_middle != self.middle {
            events[count] = Some(if new_middle {
                PointerEvent::ButtonDown { button: 2 }
            } else {
                PointerEvent::ButtonUp { button: 2 }
            });
            count += 1;
            self.middle = new_middle;
        }

        (events, count)
    }
}
