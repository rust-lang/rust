//! Legacy 8259 PIC Management
//!
//! This module provides functions to disable and mask the legacy 8259 PICs.
//! Required before IOAPIC can take over interrupt routing.

use kernel::{ioport_read_u8, ioport_write_u8};

// PIC ports
const PIC1_CMD: u16 = 0x20;
const PIC1_DATA: u16 = 0x21;
const PIC2_CMD: u16 = 0xA0;
const PIC2_DATA: u16 = 0xA1;

// ICW1: Initialization Command Word 1
const ICW1_INIT: u8 = 0x10;
const ICW1_ICW4: u8 = 0x01;

// ICW4: Initialization Command Word 4
const ICW4_8086: u8 = 0x01;

/// Remap PIC vectors to avoid conflicts with CPU exceptions (0-31) and GSI ranges.
/// Master PIC: vectors 0xF0-0xF7
/// Slave PIC: vectors 0xF8-0xFF
/// Then mask all IRQs.
pub fn disable_pic() {
    // Save existing masks (for debugging)
    let _mask1 = ioport_read_u8(PIC1_DATA);
    let _mask2 = ioport_read_u8(PIC2_DATA);

    // ICW1: Start initialization sequence
    ioport_write_u8(PIC1_CMD, ICW1_INIT | ICW1_ICW4);
    io_wait();
    ioport_write_u8(PIC2_CMD, ICW1_INIT | ICW1_ICW4);
    io_wait();

    // ICW2: Set vector offsets
    ioport_write_u8(PIC1_DATA, 0xF0); // Master PIC starts at vector 0xF0
    io_wait();
    ioport_write_u8(PIC2_DATA, 0xF8); // Slave PIC starts at vector 0xF8
    io_wait();

    // ICW3: Configure cascading
    ioport_write_u8(PIC1_DATA, 0x04); // Slave on IRQ2
    io_wait();
    ioport_write_u8(PIC2_DATA, 0x02); // Cascade identity
    io_wait();

    // ICW4: 8086 mode
    ioport_write_u8(PIC1_DATA, ICW4_8086);
    io_wait();
    ioport_write_u8(PIC2_DATA, ICW4_8086);
    io_wait();

    // Mask all IRQs on both PICs
    ioport_write_u8(PIC1_DATA, 0xFF);
    io_wait();
    ioport_write_u8(PIC2_DATA, 0xFF);
    io_wait();
}

/// Small I/O delay for PIC programming
#[inline]
fn io_wait() {
    // Write to unused port 0x80 for ~1µs delay
    ioport_write_u8(0x80, 0);
}

/// Send End-of-Interrupt to legacy PIC.
/// Supports both IRQ numbers (0-15) and legacy remapped vectors (0x20-0x2F and 0xF0-0xFF).
pub fn send_eoi(irq_or_vector: u8) {
    let is_slave = if irq_or_vector >= 0xF0 {
        irq_or_vector >= 0xF8
    } else if irq_or_vector >= 0x20 {
        irq_or_vector >= 0x28
    } else {
        irq_or_vector >= 8
    };

    if is_slave {
        ioport_write_u8(PIC2_CMD, 0x20);
    }
    ioport_write_u8(PIC1_CMD, 0x20);
}
