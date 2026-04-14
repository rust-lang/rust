#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::device::DeviceKind;
use abi::display::{
    BufferHandle, BufferId, CommitFlags, CommitRequest, DisplayInfo, PlaneCommit, PlaneId,
    DISPLAY_OP_COMMIT, DISPLAY_OP_GET_INFO, DISPLAY_OP_IMPORT_BUFFER,
};
use abi::display_protocol::Rect;
use abi::pixel::PixelFormat;
use abi::syscall::vfs_flags::O_RDONLY;
use alloc::string::String;
use alloc::vec::Vec;
use petals::bmp::load_bmp;
use stem::syscall::vfs::{vfs_close, vfs_device_call, vfs_open, vfs_readdir};
use stem::{debug, error, info};

#[stem::main]
fn main(_arg: usize) -> ! {
    stem::debug!("Bloom: VFS-native compositor starting...");

    // 1. Discover card at /dev/display/cardN
    let mut card_path = None;
    for _ in 0..50 {
        if let Some(path) = find_display_card() {
            card_path = Some(path);
            break;
        }
        stem::sleep_ms(100);
    }
    let card_path = card_path.expect("No display card found after retry!");
    stem::info!("Bloom: Selected display card: {}", card_path);

    let fd = vfs_open(&card_path, O_RDONLY).expect("Failed to open display card");

    // 2. Get display info
    let mut info_payload = unsafe { core::mem::zeroed::<DisplayInfo>() };
    let info_slice = unsafe {
        core::slice::from_raw_parts_mut(
            &mut info_payload as *mut _ as *mut u8,
            core::mem::size_of::<DisplayInfo>(),
        )
    };

    match vfs_device_call(
        fd,
        DeviceKind::Display,
        DISPLAY_OP_GET_INFO,
        info_slice.as_ptr() as u64,
    ) {
        Ok(_) => {
            stem::debug!(
                "Bloom: Display {}x{} format_mask=0x{:x}",
                info_payload.preferred_mode.width,
                info_payload.preferred_mode.height,
                info_payload.supported_formats
            );
        }
        Err(e) => {
            error!("Bloom: Failed to get display info: {:?}", e);
            loop {
                stem::yield_now();
            }
        }
    }

    // 3. Load Wallpaper
    let wallpaper_path = get_wallpaper_path();
    stem::debug!("Bloom: Loading wallpaper: {}", wallpaper_path);

    let wallpaper = load_bmp(&wallpaper_path).expect("Failed to load wallpaper BMP");
    stem::debug!(
        "Bloom: Wallpaper loaded ({}x{})",
        wallpaper.width,
        wallpaper.height
    );

    // 4. Import Buffer to card
    let import_req = BufferHandle {
        fd: wallpaper.fd,
        offset: 0,
        width: wallpaper.width,
        height: wallpaper.height,
        stride: wallpaper.stride,
        format: PixelFormat::Bgra8888,
        modifier: 0,
    };

    let buffer_id = match vfs_device_call(
        fd,
        DeviceKind::Display,
        DISPLAY_OP_IMPORT_BUFFER,
        &import_req as *const _ as u64,
    ) {
        Ok(id) => BufferId(id as u32),
        Err(e) => {
            error!("Bloom: Failed to import buffer: {:?}", e);
            loop {
                stem::yield_now();
            }
        }
    };
    stem::debug!("Bloom: Buffer imported as id={:?}", buffer_id);

    // 5. Commit to screen
    let commit = PlaneCommit {
        plane_id: PlaneId(0),
        buffer_id,
        dest_rect: Rect {
            x: 0,
            y: 0,
            w: info_payload.preferred_mode.width,
            h: info_payload.preferred_mode.height,
        },
        src_rect: Rect {
            x: 0,
            y: 0,
            w: wallpaper.width,
            h: wallpaper.height,
        },
        z_order: 0,
        alpha: 255,
        _reserved: [0; 7],
    };

    let commit_req = CommitRequest {
        commit_count: 1,
        flags: CommitFlags::empty(),
        commits_ptr: &commit as *const _ as u64,
    };

    match vfs_device_call(
        fd,
        DeviceKind::Display,
        DISPLAY_OP_COMMIT,
        &commit_req as *const _ as u64,
    ) {
        Ok(_) => {
            info!("Bloom: First paint committed successfully! 🌸");
        }
        Err(e) => {
            error!("Bloom: Failed to commit: {:?}", e);
        }
    }

    stem::debug!("Bloom: Transitioning to event loop...");
    loop {
        stem::yield_now();
    }
}

fn find_display_card() -> Option<String> {
    let fd = vfs_open("/dev/display", O_RDONLY).ok()?;
    let mut buf = [0u8; 1024];
    let n = vfs_readdir(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);

    let mut offset = 0;
    while offset < n {
        let entry = &buf[offset..];
        if entry[0] == 0 {
            break;
        }

        let mut len = 0;
        while offset + len < n && buf[offset + len] != 0 {
            len += 1;
        }

        if let Ok(name) = core::str::from_utf8(&buf[offset..offset + len]) {
            if name.starts_with("card") {
                return Some(alloc::format!("/dev/display/{}", name));
            }
        }
        offset += len + 1;
    }
    None
}

fn get_wallpaper_path() -> String {
    String::from("/share/wallpapers/flower.bmp")
}
