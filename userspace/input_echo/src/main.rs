#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::hid::{
    BristleEventHeader, EventType, KeyEventPayload, PointerButtonPayload, PointerMovePayload,
};
use stem::info;
use stem::syscall::{channel_recv, vfs_fd_from_handle, vfs_poll, ChannelHandle};
use abi::syscall::{PollFd, poll_flags};

fn log_event(buf: &[u8]) {
    if buf.len() < BristleEventHeader::SIZE {
        info!("input_echo: short event ({} bytes)", buf.len());
        return;
    }

    let mut header_bytes = [0u8; BristleEventHeader::SIZE];
    header_bytes.copy_from_slice(&buf[..BristleEventHeader::SIZE]);
    let header = match BristleEventHeader::from_bytes(&header_bytes) {
        Ok(header) => header,
        Err(err) => {
            info!("input_echo: bad header: {:?}", err);
            return;
        }
    };

    let payload = &buf[BristleEventHeader::SIZE..];
    match EventType::from_raw(header.event_type) {
        Ok(EventType::KeyDown) | Ok(EventType::KeyUp) if payload.len() >= KeyEventPayload::SIZE => {
            let mut bytes = [0u8; KeyEventPayload::SIZE];
            bytes.copy_from_slice(&payload[..KeyEventPayload::SIZE]);
            let event = KeyEventPayload::from_bytes(&bytes);
            let edge = if header.event_type == EventType::KeyDown as u16 {
                "down"
            } else {
                "up"
            };
            info!(
                "input_echo: key={} edge={} mods=0x{:02x} repeat={}",
                event.key().name(),
                edge,
                event.mods,
                event.is_repeat()
            );
        }
        Ok(EventType::PointerMove) if payload.len() >= PointerMovePayload::SIZE => {
            let mut bytes = [0u8; PointerMovePayload::SIZE];
            bytes.copy_from_slice(&payload[..PointerMovePayload::SIZE]);
            let event = PointerMovePayload::from_bytes(&bytes);
            let dx = event.dx;
            let dy = event.dy;
            info!("input_echo: pointer dx={} dy={}", dx, dy);
        }
        Ok(EventType::PointerButtonDown) | Ok(EventType::PointerButtonUp)
            if payload.len() >= PointerButtonPayload::SIZE =>
        {
            let mut bytes = [0u8; PointerButtonPayload::SIZE];
            bytes.copy_from_slice(&payload[..PointerButtonPayload::SIZE]);
            let event = PointerButtonPayload::from_bytes(&bytes);
            let edge = if header.event_type == EventType::PointerButtonDown as u16 {
                "down"
            } else {
                "up"
            };
            info!("input_echo: button={} edge={}", event.button, edge);
        }
        Ok(kind) => {
            let payload_len = header.payload_len;
            info!("input_echo: event={:?} payload_len={}", kind, payload_len)
        }
        Err(err) => info!("input_echo: bad event type: {:?}", err),
    }
}

#[stem::main]
fn main(arg: usize) -> ! {
    let handle = arg as ChannelHandle;
    info!("input_echo: starting with port={}", handle);

    if handle == 0 {
        info!("input_echo: no event port provided; idling");
        loop {
            stem::sleep_ms(1000);
        }
    }

    // Bridge the channel handle to a VFS FD for FD-first polling.
    let fd = vfs_fd_from_handle(handle).unwrap_or(0);

    let mut buf = [0u8; 256];
    loop {
        let mut pollfds = [PollFd { fd: fd as i32, events: poll_flags::POLLIN, revents: 0 }];
        match vfs_poll(&mut pollfds, u64::MAX) {
            Ok(_) => match channel_recv(handle, &mut buf) {
                Ok(n) if n > 0 => log_event(&buf[..n]),
                Ok(_) => {}
                Err(err) => info!("input_echo: recv error: {:?}", err),
            },
            Err(err) => {
                info!("input_echo: wait error: {:?}", err);
                stem::sleep_ms(10);
            }
        }
    }
}
