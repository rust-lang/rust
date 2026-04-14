//! Virtio-GPU command structures
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

// Command types
pub const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x0100;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
pub const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x0102;
pub const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x0103;
pub const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x0104;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
pub const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
pub const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;

// Response types
pub const VIRTIO_GPU_RESP_OK_NODATA: u32 = 0x1100;
pub const VIRTIO_GPU_RESP_OK_DISPLAY_INFO: u32 = 0x1101;

// Formats
pub const VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM: u32 = 1;
pub const VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM: u32 = 2;
pub const VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM: u32 = 67;
pub const VIRTIO_GPU_FORMAT_R8G8B8X8_UNORM: u32 = 68;

/// Control header for all virtio-gpu commands
#[repr(C, packed)]
pub struct VirtioGpuCtrlHdr {
    pub type_: u32,
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub padding: u32,
}

/// Display info for one scanout
#[repr(C, packed)]
pub struct VirtioGpuDisplayOne {
    pub r_x: u32,
    pub r_y: u32,
    pub r_width: u32,
    pub r_height: u32,
    pub enabled: u32,
    pub flags: u32,
}

/// Response to GET_DISPLAY_INFO
#[repr(C, packed)]
pub struct VirtioGpuRespDisplayInfo {
    pub hdr: VirtioGpuCtrlHdr,
    pub pmodes: [VirtioGpuDisplayOne; 16],
}

/// Create 2D resource request
#[repr(C, packed)]
pub struct VirtioGpuResourceCreate2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

/// Attach backing memory entry
#[repr(C, packed)]
pub struct VirtioGpuMemEntry {
    pub addr: u64,
    pub length: u32,
    pub padding: u32,
}

/// Attach backing request
#[repr(C, packed)]
pub struct VirtioGpuResourceAttachBacking {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub nr_entries: u32,
    // Followed by VirtioGpuMemEntry array
}

/// Set scanout request
#[repr(C, packed)]
pub struct VirtioGpuSetScanout {
    pub hdr: VirtioGpuCtrlHdr,
    pub r_x: u32,
    pub r_y: u32,
    pub r_width: u32,
    pub r_height: u32,
    pub scanout_id: u32,
    pub resource_id: u32,
}

/// Transfer to host request
#[repr(C, packed)]
pub struct VirtioGpuTransferToHost2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub r_x: u32,
    pub r_y: u32,
    pub r_width: u32,
    pub r_height: u32,
    pub offset: u64,
    pub resource_id: u32,
    pub padding: u32,
}

/// Flush resource request
#[repr(C, packed)]
pub struct VirtioGpuResourceFlush {
    pub hdr: VirtioGpuCtrlHdr,
    pub r_x: u32,
    pub r_y: u32,
    pub r_width: u32,
    pub r_height: u32,
    pub resource_id: u32,
    pub padding: u32,
}

// ============================================================================
// 3D/Virgl Commands (0x0200 range)
// ============================================================================

// 3D command types
pub const VIRTIO_GPU_CMD_CTX_CREATE: u32 = 0x0200;
pub const VIRTIO_GPU_CMD_CTX_DESTROY: u32 = 0x0201;
pub const VIRTIO_GPU_CMD_CTX_ATTACH_RESOURCE: u32 = 0x0202;
pub const VIRTIO_GPU_CMD_CTX_DETACH_RESOURCE: u32 = 0x0203;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_3D: u32 = 0x0204;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_3D: u32 = 0x0205;
pub const VIRTIO_GPU_CMD_TRANSFER_FROM_HOST_3D: u32 = 0x0206;
pub const VIRTIO_GPU_CMD_SUBMIT_3D: u32 = 0x0207;

// 3D resource targets (gallium pipe_texture_target)
pub const PIPE_BUFFER: u32 = 0;
pub const PIPE_TEXTURE_1D: u32 = 1;
pub const PIPE_TEXTURE_2D: u32 = 2;
pub const PIPE_TEXTURE_3D: u32 = 3;
pub const PIPE_TEXTURE_CUBE: u32 = 4;
pub const PIPE_TEXTURE_RECT: u32 = 5;
pub const PIPE_TEXTURE_1D_ARRAY: u32 = 6;
pub const PIPE_TEXTURE_2D_ARRAY: u32 = 7;

// 3D resource bind flags (gallium pipe_bind)
pub const PIPE_BIND_DEPTH_STENCIL: u32 = 1 << 0;
pub const PIPE_BIND_RENDER_TARGET: u32 = 1 << 1;
pub const PIPE_BIND_SAMPLER_VIEW: u32 = 1 << 3;
pub const PIPE_BIND_VERTEX_BUFFER: u32 = 1 << 4;
pub const PIPE_BIND_INDEX_BUFFER: u32 = 1 << 5;
pub const PIPE_BIND_CONSTANT_BUFFER: u32 = 1 << 6;
pub const PIPE_BIND_SCANOUT: u32 = 1 << 14;

/// Create 3D context request
#[repr(C, packed)]
pub struct VirtioGpuCtxCreate {
    pub hdr: VirtioGpuCtrlHdr,
    pub nlen: u32,
    pub context_init: u32, // capset id to use for context init
    pub debug_name: [u8; 64],
}

/// Destroy 3D context request
#[repr(C, packed)]
pub struct VirtioGpuCtxDestroy {
    pub hdr: VirtioGpuCtrlHdr,
}

/// Attach resource to context request
#[repr(C, packed)]
pub struct VirtioGpuCtxResource {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub padding: u32,
}

/// Create 3D resource request
#[repr(C, packed)]
pub struct VirtioGpuResourceCreate3d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub target: u32, // PIPE_TEXTURE_*
    pub format: u32, // virgl format
    pub bind: u32,   // PIPE_BIND_* flags
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
    pub padding: u32,
}

/// Transfer to host 3D request
#[repr(C, packed)]
pub struct VirtioGpuTransferToHost3d {
    pub hdr: VirtioGpuCtrlHdr,
    pub box_x: u32,
    pub box_y: u32,
    pub box_z: u32,
    pub box_w: u32,
    pub box_h: u32,
    pub box_d: u32,
    pub offset: u64,
    pub resource_id: u32,
    pub level: u32,
    pub stride: u32,
    pub layer_stride: u32,
}

/// Submit 3D command buffer request
#[repr(C, packed)]
pub struct VirtioGpuCmdSubmit3d {
    pub hdr: VirtioGpuCtrlHdr,
    pub size: u32,
    pub padding: u32,
    // Followed by `size` bytes of virgl command stream
}
