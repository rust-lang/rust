#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::Texture;
use abi::syscall::vfs_flags::O_RDONLY;
use alloc::vec;
use alloc::vec::Vec;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

pub fn load_bmp(path: &str) -> Option<Texture> {
    let fd = vfs_open(path, O_RDONLY).ok()?;

    let mut header = [0u8; 54];
    if vfs_read(fd, &mut header).ok()? < 54 {
        let _ = vfs_close(fd);
        return None;
    }

    if &header[0..2] != b"BM" {
        let _ = vfs_close(fd);
        return None;
    }

    let offset = u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize;
    let width = i32::from_le_bytes([header[18], header[19], header[20], header[21]]) as u32;
    let height = i32::from_le_bytes([header[22], header[23], header[24], header[25]]);
    let bpp = u16::from_le_bytes([header[28], header[29]]);

    let is_bottom_up = height > 0;
    let abs_height = height.abs() as u32;

    let mut texture = Texture::new(path, width, abs_height, 4)?;
    let tex_slice = texture.as_slice_mut();

    // Seek to pixels
    // We don't have vfs_seek easily in some contexts via stem, but we can just read if small,
    // or we've already read 54 bytes.
    let mut remaining_to_skip = offset - 54;
    let mut skip_buf = [0u8; 1024];
    while remaining_to_skip > 0 {
        let to_read = remaining_to_skip.min(1024);
        let n = vfs_read(fd, &mut skip_buf[..to_read]).ok()?;
        if n == 0 {
            break;
        }
        remaining_to_skip -= n;
    }

    if bpp == 24 {
        let row_padding = (4 - (width * 3) % 4) % 4;
        let mut row_buf = vec![0u8; (width * 3 + row_padding) as usize];

        for y in 0..abs_height {
            if vfs_read(fd, &mut row_buf).ok()? < (width * 3) as usize {
                break;
            }

            let target_y = if is_bottom_up { abs_height - 1 - y } else { y };
            let row_offset = (target_y * width) as usize;

            for x in 0..width {
                let b = row_buf[(x * 3) as usize];
                let g = row_buf[(x * 3 + 1) as usize];
                let r = row_buf[(x * 3 + 2) as usize];
                tex_slice[row_offset + x as usize] =
                    (0xFF << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            }
        }
    } else if bpp == 32 {
        for y in 0..abs_height {
            let target_y = if is_bottom_up { abs_height - 1 - y } else { y };
            let row_offset = (target_y * width) as usize;
            let row_ptr = unsafe { tex_slice.as_mut_ptr().add(row_offset) as *mut u8 };
            let row_len = (width * 4) as usize;
            let row_slice = unsafe { core::slice::from_raw_parts_mut(row_ptr, row_len) };

            if vfs_read(fd, row_slice).ok()? < row_len {
                break;
            }
        }
    }

    let _ = vfs_close(fd);
    Some(texture)
}
