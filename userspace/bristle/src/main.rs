//! Bristle: Unified HID Broker
//!
//! The sole input authority. Drivers send raw reports, apps receive
//! normalized events. Apps never see scancodes, drivers never see apps.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::hid::{
    BRISTLE_EVENT_MAGIC, BRISTLE_EVENT_VERSION, BristleEventHeader, EventType, Key,
    KeyEventPayload, PointerButtonPayload, PointerMovePayload,
};
use abi::syscall::vfs_flags::{O_CREAT, O_RDWR, O_TRUNC};
use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_write};
use stem::syscall::{ChannelHandle, channel_recv, channel_send_all};
use stem::{debug, info};

fn ensure_session_roots() {
    let _ = vfs_mkdir("/session");
    let _ = vfs_mkdir("/session/seat0");
    let _ = vfs_mkdir("/session/seat0/keyboard");
    let _ = vfs_mkdir("/session/seat0/pointer");
}

fn update_active_ui(target: &str) {
    if let Ok(fd) = vfs_open("/session/active_ui", O_RDWR | O_CREAT | O_TRUNC) {
        let _ = vfs_write(fd, target.as_bytes());
        let _ = vfs_close(fd);
        stem::debug!("bristle: active_ui set to '{}'", target);
    }
}

fn get_active_ui() -> alloc::string::String {
    use stem::syscall::vfs::{vfs_read, vfs_stat};
    if let Ok(fd) = vfs_open("/session/active_ui", abi::syscall::vfs_flags::O_RDONLY) {
        if let Ok(stat) = vfs_stat(fd) {
            let size = stat.size as usize;
            let mut buf = alloc::vec::Vec::with_capacity(size as usize);
            buf.resize(size as usize, 0);
            if let Ok(n) = vfs_read(fd, &mut buf) {
                buf.truncate(n);
                let _ = vfs_close(fd);
                return alloc::string::String::from_utf8_lossy(&buf)
                    .trim()
                    .to_string();
            }
        }
        let _ = vfs_close(fd);
    }
    "terminal".to_string()
}

#[stem::main]
fn main(packed_handles: usize) -> ! {
    let packed = packed_handles as u64;
    let kbd_read = ((packed >> 48) & 0xFFFF) as ChannelHandle;
    let mouse_read = ((packed >> 32) & 0xFFFF) as ChannelHandle;
    let bloom_evt_write = ((packed >> 16) & 0xFFFF) as ChannelHandle;
    let evt_input_echo_write = (packed & 0xFFFF) as ChannelHandle;

    stem::debug!(
        "bristle: online (kbd={}, mouse={}, bloom_evt={}, input_echo={})",
        kbd_read,
        mouse_read,
        bloom_evt_write,
        evt_input_echo_write
    );

    ensure_session_roots();
    update_active_ui("bloom");

    let mut recv_buf = [0u8; 128];
    let mut event_accum = [0u8; 64];
    let mut accum_len = 0usize;
    let mut drop_counter: u32 = 0;

    let mut ws = stem::wait_set::WaitSet::new();
    let mut kbd_tok = None;
    let mut mouse_tok = None;

    if kbd_read != 0 {
        kbd_tok = ws.add_port_readable(kbd_read as u64).ok();
    }
    if mouse_read != 0 {
        mouse_tok = ws.add_port_readable(mouse_read as u64).ok();
    }

    loop {
        let events = match ws.wait(None::<stem::time::Duration>) {
            Ok(evs) => evs,
            Err(_) => {
                stem::time::sleep_ms(10);
                continue;
            }
        };

        for ev in events {
            if !ev.is_readable() {
                continue;
            }

            let ready_handle = if Some(ev.token()) == kbd_tok {
                kbd_read
            } else if Some(ev.token()) == mouse_tok {
                mouse_read
            } else {
                continue;
            };

            if let Ok(n) = channel_recv(ready_handle, &mut recv_buf) {
                if n > 0 {
                    let mut cursor = 0;
                    while cursor < n {
                        let to_copy = (n - cursor).min(64 - accum_len);
                        event_accum[accum_len..accum_len + to_copy]
                            .copy_from_slice(&recv_buf[cursor..cursor + to_copy]);
                        accum_len += to_copy;
                        cursor += to_copy;

                        while accum_len >= BristleEventHeader::SIZE {
                            let mut header_bytes = [0u8; BristleEventHeader::SIZE];
                            header_bytes.copy_from_slice(&event_accum[..BristleEventHeader::SIZE]);

                            if let Ok(header) = BristleEventHeader::from_bytes(&header_bytes) {
                                let total_len =
                                    BristleEventHeader::SIZE + header.payload_len as usize;
                                if accum_len >= total_len {
                                    let event_bytes = &event_accum[..total_len];

                                    // Hotkey handling
                                    if header.event_type == EventType::KeyDown as u16
                                        && header.payload_len >= 4
                                    {
                                        let mut p = [0u8; 4];
                                        p.copy_from_slice(&event_bytes[20..24]);
                                        let payload = KeyEventPayload::from_bytes(&p);

                                        match payload.key() {
                                            Key::F2 => {
                                                info!("bristle: F2 pressed - dumping tasks...");
                                                stem::syscall::task_dump();
                                            }
                                            Key::Delete
                                                if payload.mods().has_ctrl()
                                                    && payload.mods().has_alt() =>
                                            {
                                                info!("bristle: Ctrl+Alt+Del - rebooting...");
                                                stem::syscall::reboot();
                                            }
                                            Key::F1 => {
                                                update_active_ui("bloom");
                                            }
                                            Key::F12 => {
                                                update_active_ui("terminal");
                                            }
                                            _ => {}
                                        }
                                    }

                                    // Send to consumers
                                    if channel_send_all(bloom_evt_write, event_bytes).is_err() {
                                        drop_counter += 1;
                                    }

                                    if evt_input_echo_write != 0
                                        && channel_send_all(evt_input_echo_write, event_bytes)
                                            .is_err()
                                    {
                                        drop_counter += 1;
                                    }

                                    // Shift remaining
                                    accum_len -= total_len;
                                    if accum_len > 0 {
                                        event_accum
                                            .copy_within(total_len..total_len + accum_len, 0);
                                    }
                                } else {
                                    break;
                                }
                            } else {
                                // Invalid header, resync
                                accum_len -= 1;
                                if accum_len > 0 {
                                    event_accum.copy_within(1..1 + accum_len, 0);
                                }
                            }
                        }
                    }
                }
            }
        }

        if drop_counter > 0 && drop_counter % 100 == 0 {
            info!("bristle: dropped {} events (port full)", drop_counter);
        }
    }
}
