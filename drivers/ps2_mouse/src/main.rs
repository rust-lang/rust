//! PS/2 Mouse Driver (Interrupt-driven)
//!
//! Subscribes to IRQ12 via IOAPIC, reads mouse packets on interrupt, sends to Bristle.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use stem::syscall::{ChannelHandle, channel_send_all, ioport_read, ioport_write, irq_subscribe};
use stem::{debug, info};

const PS2_DATA: usize = 0x60;
const PS2_STATUS: usize = 0x64;
const PS2_CMD: usize = 0x64;

const STATUS_OUTPUT_FULL: usize = 0x01;
const STATUS_AUX_DATA: usize = 0x20;

const CMD_ENABLE_AUX: u8 = 0xA8;
const CMD_READ_CFG: u8 = 0x20;
const CMD_WRITE_CFG: u8 = 0x60;
const CMD_WRITE_AUX: u8 = 0xD4;
const MOUSE_ENABLE: u8 = 0xF4;

/// IRQ12 vector (mouse) - legacy IRQ12 maps to vector 0x2C after IOAPIC remap
const MOUSE_VECTOR: u8 = 0x2C;
const POLLING_INTERVAL_MS: u64 = 2;

fn wait_input_empty() {
    for _ in 0..10000 {
        if ioport_read(PS2_STATUS, 1) & 0x02 == 0 {
            return;
        }
        stem::yield_now();
    }
}

fn flush_output_buffer() {
    // Drain up to 16 bytes of garbage
    for _ in 0..16 {
        if ioport_read(PS2_STATUS, 1) & STATUS_OUTPUT_FULL != 0 {
            let b = ioport_read(PS2_DATA, 1);
            debug!("ps2_mouse: flushed garbage byte: 0x{:02x}", b);
        } else {
            break;
        }
        stem::yield_now();
    }
}

fn read_data_filtered(expect_aux: bool, label: &str) -> Option<u8> {
    let discarded_aux: u32 = 0;
    let discarded_non_aux: u32 = 0;
    for _ in 0..20_000 {
        let status = ioport_read(PS2_STATUS, 1);
        if status & STATUS_OUTPUT_FULL == 0 {
            stem::yield_now();
            continue;
        }
        let is_aux = (status & STATUS_AUX_DATA) != 0;

        if is_aux != expect_aux {
            // Leave bytes for the matching side of the shared controller.
            stem::yield_now();
            continue;
        }
        let byte = ioport_read(PS2_DATA, 1) as u8;
        return Some(byte);
    }
    debug!(
        "ps2_mouse: timed out waiting for {} (discarded_aux={}, discarded_non_aux={})",
        label, discarded_aux, discarded_non_aux
    );
    None
}

fn read_controller_config() -> u8 {
    wait_input_empty();
    // Flush any pending data (e.g. key scancodes) before asking for config
    flush_output_buffer();
    ioport_write(PS2_CMD, CMD_READ_CFG as usize, 1);
    if let Some(val) = read_data_filtered(false, "controller cfg") {
        return val;
    }
    // Fallback if we keep getting garbage
    debug!("ps2_mouse: read_cfg failed, assuming default safe config (0x47)");
    0x47 // IRQ1, IRQ12, SysFlag, Translation
}

fn write_controller_config(cfg: u8) {
    wait_input_empty();
    ioport_write(PS2_CMD, CMD_WRITE_CFG as usize, 1);
    wait_input_empty();
    ioport_write(PS2_DATA, cfg as usize, 1);
}

fn send_aux_byte(byte: u8) {
    wait_input_empty();
    ioport_write(PS2_CMD, CMD_WRITE_AUX as usize, 1);
    wait_input_empty();
    ioport_write(PS2_DATA, byte as usize, 1);
}

fn init_mouse() {
    debug!("ps2_mouse: enabling aux port");

    // Clear any initial garbage
    flush_output_buffer();

    // Enable aux port
    wait_input_empty();
    ioport_write(PS2_CMD, CMD_ENABLE_AUX as usize, 1);
    for _ in 0..10 {
        stem::yield_now();
    }

    // Ensure IRQ12 is enabled (Bit 1) and Mouse Disabled (Bit 5) is CLEARED.
    // Bit 5: 1 = Mouse Disabled, 0 = Mouse Enabled.
    let cfg = read_controller_config();

    // Force: Set Bit 1 (IRQ12), Clear Bit 5 (Mouse Disable)
    let new_cfg = (cfg | 0x02) & !0x20;

    if new_cfg != cfg {
        write_controller_config(new_cfg);
        debug!(
            "ps2_mouse: updated controller cfg 0x{:02x} -> 0x{:02x}",
            cfg, new_cfg
        );
    } else {
        debug!("ps2_mouse: controller cfg already correct (0x{:02x})", cfg);
    }

    // Reset mouse (0xFF)
    debug!("ps2_mouse: sending RESET (0xFF)");
    send_aux_byte(0xFF);
    let ack = read_data_filtered(true, "reset ACK (0xfa)").unwrap_or(0);
    if ack == 0xFA {
        debug!("ps2_mouse: reset ACK received (0xfa)");
        let bat = read_data_filtered(true, "BAT byte (0xAA)").unwrap_or(0);
        let id = read_data_filtered(true, "Device ID (0x00)").unwrap_or(1);
        debug!(
            "ps2_mouse: BAT passed (0x{:02x}), ID 0x{:02x} confirmed",
            bat, id
        );
    }

    debug!("ps2_mouse: setting sample rate (100)");
    send_aux_byte(0xF3);
    read_data_filtered(true, "sample rate ACK");
    send_aux_byte(100);
    read_data_filtered(true, "sample rate set ACK");

    debug!("ps2_mouse: setting resolution (3)");
    send_aux_byte(0xE8);
    read_data_filtered(true, "resolution ACK");
    send_aux_byte(3);
    read_data_filtered(true, "resolution set ACK");

    send_aux_byte(0xE9);
    let _s_ack = read_data_filtered(true, "status request ACK");
    let b1 = read_data_filtered(true, "status byte 1").unwrap_or(0);
    let b2 = read_data_filtered(true, "status byte 2").unwrap_or(0);
    let b3 = read_data_filtered(true, "status byte 3").unwrap_or(0);
    debug!(
        "ps2_mouse: status result = Some({}) Some({}) Some({})",
        b1, b2, b3
    );

    // Bit 5 indicates Enable/Disable status (1 = Enabled, 0 = Disabled).
    // If it is 0, data reporting is disabled, so we must enable it.
    if b1 & 0x20 == 0 {
        // Enable mouse data reporting (0xF4)
        debug!("ps2_mouse: sending enable command (0xF4)");
        send_aux_byte(MOUSE_ENABLE);
        let e_ack = read_data_filtered(true, "enable ACK (0xFA)").unwrap_or(0);
        debug!("ps2_mouse: enable ACK received (0x{:02x})", e_ack);
    } else {
        debug!("ps2_mouse: already enabled, skipping 0xF4 command");
    }

    stem::sleep_ms(100);

    // Drain any lingering response bytes.
    for _ in 0..10 {
        if ioport_read(PS2_STATUS, 1) & STATUS_OUTPUT_FULL != 0 {
            let byte = ioport_read(PS2_DATA, 1) as u8;
            debug!("ps2_mouse: drained 0x{:02x}", byte);
        }
        stem::sleep_ms(10);
    }

    debug!("ps2_mouse: init done");
}

#[stem::main]
fn main(raw_write_handle: usize) -> ! {
    let handle = raw_write_handle as ChannelHandle;

    stem::debug!("ps2_mouse: online (handle={})", handle);

    init_mouse();

    // Subscribe to mouse interrupt
    match irq_subscribe(MOUSE_VECTOR) {
        Ok(()) => debug!(
            "ps2_mouse: subscribed to IRQ12 (vector 0x{:02x})",
            MOUSE_VECTOR
        ),
        Err(e) => {
            debug!(
                "ps2_mouse: IRQ subscribe failed ({:?}), falling back to polling",
                e
            );
            polling_loop(handle);
        }
    }
    // Keep servicing the controller via polling even when IRQ12 subscription succeeds.
    // This avoids a dead cursor on platforms where legacy PS/2 interrupts never wake userspace.
    polling_loop(handle);
}

mod mouse;

use abi::hid::{
    BRISTLE_EVENT_MAGIC, BRISTLE_EVENT_VERSION, BristleEventHeader, EventType,
    PointerButtonPayload, PointerMovePayload,
};
use mouse::{MouseState, PointerEvent};

fn send_mouse_events(
    handle: ChannelHandle,
    state: &mut MouseState,
    packet: &[u8; 3],
    drop_counter: &mut u32,
) {
    let (events, count) = state.process_packet(packet);
    for i in 0..count {
        if let Some(evt) = events[i] {
            let timestamp_ns = stem::monotonic_ns();
            let mut buf = [0u8; 24]; // Max size is header + 4 byte payload
            let len;

            match evt {
                PointerEvent::Move { dx, dy } => {
                    let header = BristleEventHeader {
                        magic: BRISTLE_EVENT_MAGIC,
                        version: BRISTLE_EVENT_VERSION,
                        event_type: EventType::PointerMove as u16,
                        timestamp_ns,
                        payload_len: PointerMovePayload::SIZE as u32,
                    };
                    let payload = PointerMovePayload { dx, dy };
                    buf[0..20].copy_from_slice(&header.to_bytes());
                    buf[20..24].copy_from_slice(&payload.to_bytes());
                    len = 24;
                }
                PointerEvent::ButtonDown { button } => {
                    let header = BristleEventHeader {
                        magic: BRISTLE_EVENT_MAGIC,
                        version: BRISTLE_EVENT_VERSION,
                        event_type: EventType::PointerButtonDown as u16,
                        timestamp_ns,
                        payload_len: PointerButtonPayload::SIZE as u32,
                    };
                    let payload = PointerButtonPayload { button, _pad: 0 };
                    buf[0..20].copy_from_slice(&header.to_bytes());
                    buf[20..22].copy_from_slice(&payload.to_bytes());
                    len = 22;
                }
                PointerEvent::ButtonUp { button } => {
                    let header = BristleEventHeader {
                        magic: BRISTLE_EVENT_MAGIC,
                        version: BRISTLE_EVENT_VERSION,
                        event_type: EventType::PointerButtonUp as u16,
                        timestamp_ns,
                        payload_len: PointerButtonPayload::SIZE as u32,
                    };
                    let payload = PointerButtonPayload { button, _pad: 0 };
                    buf[0..20].copy_from_slice(&header.to_bytes());
                    buf[20..22].copy_from_slice(&payload.to_bytes());
                    len = 22;
                }
            }
            if len > 0 && channel_send_all(handle, &buf[..len]).is_err() {
                *drop_counter = drop_counter.wrapping_add(1);
                if *drop_counter <= 4 || *drop_counter % 100 == 0 {
                    info!(
                        "ps2_mouse: dropped {} mouse events because raw input port {} is full",
                        *drop_counter, handle
                    );
                }
            }
        }
    }
}

/// Drain all pending mouse data and assemble packets
fn drain_mouse_data(
    handle: ChannelHandle,
    state: &mut MouseState,
    packet: &mut [u8; 3],
    idx: &mut usize,
    drop_counter: &mut u32,
) {
    for _ in 0..16 {
        let status = ioport_read(PS2_STATUS, 1);

        if status & STATUS_OUTPUT_FULL == 0 {
            break;
        }

        if status & STATUS_AUX_DATA != 0 {
            let byte = ioport_read(PS2_DATA, 1) as u8;

            // First byte must have bit 3 set (sync)
            if *idx == 0 && (byte & 0x08) == 0 {
                continue;
            }

            packet[*idx] = byte;
            *idx += 1;

            if *idx == 3 {
                send_mouse_events(handle, state, packet, drop_counter);
                *idx = 0;
            }
        } else {
            // Shared i8042 controller: leave keyboard bytes for ps2_kbd.
            // Consuming them here makes keyboard input appear dead.
            break;
        }
    }
}

/// Fallback polling loop
fn polling_loop(handle: ChannelHandle) -> ! {
    stem::debug!(
        "ps2_mouse: using cooperative polling loop ({}ms interval)",
        POLLING_INTERVAL_MS
    );

    let mut packet = [0u8; 3];
    let mut idx = 0usize;
    let mut mouse_state = MouseState::new();
    let mut drop_counter = 0u32;

    loop {
        let status = ioport_read(PS2_STATUS, 1);

        if status & STATUS_OUTPUT_FULL != 0 {
            if status & STATUS_AUX_DATA != 0 {
                drain_mouse_data(
                    handle,
                    &mut mouse_state,
                    &mut packet,
                    &mut idx,
                    &mut drop_counter,
                );
            } else {
                // Leave keyboard bytes queued for ps2_kbd.
                stem::sleep_ms(1);
            }
        } else {
            stem::sleep_ms(POLLING_INTERVAL_MS);
        }
    }
}
