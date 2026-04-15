#![no_std]
#![no_main]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use abi::display_driver_protocol::{FB_INFO_PAYLOAD_SIZE, FbInfoPayload};
use abi::syscall::vfs_flags::{O_RDONLY, O_WRONLY};
use petals::bmp::load_bmp;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_write};
use stem::{error, info};

#[stem::main]
fn main(_arg: usize) -> ! {
    stem::debug!("Bloom: VFS-native compositor starting...");

    // 1. Discover the boot framebuffer at /dev/fb0.
    let mut fb_info = None;
    for _ in 0..50 {
        if let Some(info) = read_bootfb_info() {
            fb_info = Some(info);
            break;
        }
        stem::sleep_ms(100);
    }
    let fb_info = fb_info.expect("No /dev/fb0 found after retry!");
    stem::info!(
        "Bloom: Using boot framebuffer {}x{} stride={} bpp={} format={}",
        fb_info.width,
        fb_info.height,
        fb_info.stride,
        fb_info.bpp,
        fb_info.format
    );

    if fb_info.bpp != 32 || fb_info.stride < fb_info.width * 4 {
        error!(
            "Bloom: Unsupported boot framebuffer layout: width={} height={} stride={} bpp={}",
            fb_info.width, fb_info.height, fb_info.stride, fb_info.bpp
        );
        loop {
            stem::yield_now();
        }
    }

    // 2. Load wallpaper
    let wallpaper_path = get_wallpaper_path();
    stem::debug!("Bloom: Loading wallpaper: {}", wallpaper_path);

    let mut wallpaper = load_bmp(&wallpaper_path).expect("Failed to load wallpaper BMP");
    stem::debug!("Bloom: Wallpaper loaded ({}x{})", wallpaper.width, wallpaper.height);

    // 3. Compose a framebuffer-sized image and write it directly to /dev/fb0.
    let frame = compose_wallpaper_frame(&mut wallpaper, &fb_info);
    let frame_bytes = unsafe {
        core::slice::from_raw_parts(
            frame.as_ptr() as *const u8,
            frame.len() * core::mem::size_of::<u32>(),
        )
    };
    let fd = vfs_open("/dev/fb0", O_WRONLY).expect("Failed to open /dev/fb0 for writing");
    match vfs_write(fd, frame_bytes) {
        Ok(n) if n == frame_bytes.len() => {
            info!("Bloom: First paint committed successfully!");
        }
        Ok(n) => {
            error!("Bloom: Short write to /dev/fb0: wrote {} of {} bytes", n, frame_bytes.len());
        }
        Err(e) => error!("Bloom: Failed to write /dev/fb0: {:?}", e),
    }
    let _ = vfs_close(fd);

    stem::debug!("Bloom: Transitioning to event loop...");
    loop {
        stem::yield_now();
    }
}

fn read_bootfb_info() -> Option<FbInfoPayload> {
    let fd = vfs_open("/dev/fb0", O_RDONLY).ok()?;
    let mut payload = FbInfoPayload {
        device_handle: 0,
        width: 0,
        height: 0,
        stride: 0,
        bpp: 0,
        format: 0,
        _reserved: 0,
    };
    let slice = unsafe {
        core::slice::from_raw_parts_mut(&mut payload as *mut _ as *mut u8, FB_INFO_PAYLOAD_SIZE)
    };
    let n = vfs_read(fd, slice).ok()?;
    let _ = vfs_close(fd);
    if n < FB_INFO_PAYLOAD_SIZE || payload.width == 0 || payload.height == 0 || payload.stride == 0
    {
        return None;
    }
    Some(payload)
}

fn compose_wallpaper_frame(wallpaper: &mut petals::Texture, fb_info: &FbInfoPayload) -> Vec<u32> {
    let dst_stride_pixels = (fb_info.stride / 4) as usize;
    let dst_len = (fb_info.height as usize) * dst_stride_pixels;
    let mut frame = alloc::vec![0xFF000000u32; dst_len];

    let src_stride_pixels = (wallpaper.stride / 4) as usize;
    let src_w = wallpaper.width as usize;
    let src_h = wallpaper.height as usize;
    let src = wallpaper.as_slice_mut();
    let dst_w = fb_info.width as usize;
    let dst_h = fb_info.height as usize;

    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return frame;
    }

    let scale_x = (dst_w as u64 * 1_000_000) / src_w as u64;
    let scale_y = (dst_h as u64 * 1_000_000) / src_h as u64;
    let scale = scale_x.min(scale_y).max(1);

    let scaled_w = ((src_w as u64 * scale) / 1_000_000) as usize;
    let scaled_h = ((src_h as u64 * scale) / 1_000_000) as usize;
    let offset_x = (dst_w.saturating_sub(scaled_w)) / 2;
    let offset_y = (dst_h.saturating_sub(scaled_h)) / 2;

    for dy in 0..scaled_h {
        let sy = ((dy as u64 * src_h as u64) / scaled_h as u64) as usize;
        let dst_row = (offset_y + dy) * dst_stride_pixels;
        let src_row = sy * src_stride_pixels;
        for dx in 0..scaled_w {
            let sx = ((dx as u64 * src_w as u64) / scaled_w as u64) as usize;
            frame[dst_row + offset_x + dx] = src[src_row + sx];
        }
    }

    frame
}

fn get_wallpaper_path() -> String {
    String::from("/share/wallpapers/flower.bmp")
}
