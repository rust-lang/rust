#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
pub const FB_INFO_PAYLOAD_SIZE: usize = 24;
use abi::display::{BufferId, CommitRequest, DisplayInfo, PlaneCommit, PlaneId};
use abi::errors::{Errno, SysResult};
use abi::pixel::PixelFormat;
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use alloc::collections::BTreeMap;
use stem::syscall::{vfs_close, vfs_open, vfs_read, vm_map, vm_unmap};
use stem::{debug, info};

/// HW Framebuffer description
pub struct Framebuffer {
    pub base: *mut u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub bpp: u32,
}

/// A buffer imported from userspace and mapped into driver memory
pub struct MappedBuffer {
    pub ptr: *mut u8,
    pub size: usize,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PixelFormat,
}

pub struct BootFbDriver {
    pub fb: Framebuffer,
    pub buffers: BTreeMap<BufferId, MappedBuffer>,
    pub next_buffer_id: u32,
}

impl BootFbDriver {
    pub fn new() -> Option<Self> {
        let fb = find_framebuffer()?;
        Some(Self {
            fb,
            buffers: BTreeMap::new(),
            next_buffer_id: 1,
        })
    }

    pub fn get_info(&self) -> DisplayInfo {
        DisplayInfo {
            card_id: 0,
            preferred_mode: abi::display::DisplayMode {
                width: self.fb.width,
                height: self.fb.height,
                refresh_mhz: 60000,
            },
            plane_count: 1,
            max_buffers: 32,
            supported_formats: 1 << (PixelFormat::Bgra8888 as u8),
            caps: abi::display::DisplayCaps::empty(),
        }
    }

    pub fn import_buffer(&mut self, handle: &abi::display::BufferHandle) -> SysResult<BufferId> {
        let size = (handle.height as usize) * (handle.stride as usize);
        let req = VmMapReq {
            addr_hint: 0,
            len: size,
            prot: VmProt::READ | VmProt::USER,
            flags: VmMapFlags::PRIVATE,
            backing: VmBacking::File {
                fd: handle.fd,
                offset: handle.offset,
            },
        };

        let resp = vm_map(&req).map_err(|_| Errno::ENOMEM)?;
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;

        self.buffers.insert(
            id,
            MappedBuffer {
                ptr: resp.addr as *mut u8,
                size,
                width: handle.width,
                height: handle.height,
                stride: handle.stride,
                format: handle.format,
            },
        );

        debug!(
            "display_bootfb: imported buffer {} ({}x{} @ {:p})",
            id.0, handle.width, handle.height, resp.addr as *mut u8
        );
        Ok(id)
    }

    pub fn release_buffer(&mut self, id: BufferId) -> SysResult<()> {
        if let Some(buf) = self.buffers.remove(&id) {
            let _ = vm_unmap(buf.ptr as usize, buf.size);
            Ok(())
        } else {
            Err(Errno::ENOENT)
        }
    }

    pub fn commit(&mut self, req: &CommitRequest) -> SysResult<()> {
        // Bootfb only supports a single primary plane.
        // We find the first plane commit that targets the primary plane (Id 0).
        for plane in req.planes() {
            if plane.plane_id == PlaneId(0) {
                return self.blit_primary(plane);
            }
        }
        Ok(())
    }

    fn blit_primary(&mut self, commit: &PlaneCommit) -> SysResult<()> {
        let buffer = self.buffers.get(&commit.buffer_id).ok_or(Errno::ENOENT)?;

        // Simple full-frame blit for now.
        // TODO: Respect commit.src_rect and commit.dst_rect.

        let pitch_bpp = if self.fb.width > 0 {
            (self.fb.stride / self.fb.width) as usize
        } else {
            0
        };
        let mut bpp = (self.fb.bpp / 8) as usize;
        if pitch_bpp >= bpp && pitch_bpp > 0 {
            bpp = pitch_bpp;
        }

        let h = self.fb.height.min(buffer.height) as usize;
        let row_bytes = (self.fb.width.min(buffer.width) as usize) * bpp;

        for row in 0..h {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buffer.ptr.add(row * buffer.stride as usize),
                    self.fb.base.add(row * self.fb.stride as usize),
                    row_bytes,
                );
            }
        }

        Ok(())
    }
}

fn find_framebuffer() -> Option<Framebuffer> {
    use abi::display_driver_protocol::FbInfoPayload;
    use abi::syscall::vfs_flags::O_RDONLY;

    debug!("display_bootfb: probing /dev/fb0...");
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
    if n < FB_INFO_PAYLOAD_SIZE {
        let _ = vfs_close(fd);
        return None;
    }

    let byte_len = (payload.height as usize) * (payload.stride as usize);
    let req = VmMapReq {
        addr_hint: 0,
        len: byte_len,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::empty(),
        backing: VmBacking::File { fd, offset: 0 },
    };

    let resp = match vm_map(&req) {
        Ok(resp) => resp,
        Err(e) => {
            debug!("display_bootfb: failed to map /dev/fb0: {:?}", e);
            let _ = vfs_close(fd);
            return None;
        }
    };

    let _ = vfs_close(fd);

    Some(Framebuffer {
        base: resp.addr as *mut u8,
        width: payload.width,
        height: payload.height,
        stride: payload.stride,
        bpp: payload.bpp,
    })
}
