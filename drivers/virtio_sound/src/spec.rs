//! VirtIO Sound Device Specification (v1.2)
//! https://docs.oasis-open.org/virtio/virtio/v1.2/csd01/virtio-v1.2-csd01.pdf
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::vec::Vec;
use core::mem::size_of;

// Feature bits
pub const VIRTIO_SND_F_CTLS: u32 = 0; // Device supports control elements

// Queue indices
pub const VIRTIO_SND_VQ_CONTROL: u16 = 0;
pub const VIRTIO_SND_VQ_EVENT: u16 = 1;
pub const VIRTIO_SND_VQ_TX: u16 = 2;
pub const VIRTIO_SND_VQ_RX: u16 = 3;

// Request types
pub const VIRTIO_SND_R_JACK_INFO: u32 = 1;
pub const VIRTIO_SND_R_JACK_REMAP: u32 = 2;
pub const VIRTIO_SND_R_PCM_INFO: u32 = 0x0100;
pub const VIRTIO_SND_R_PCM_SET_PARAMS: u32 = 0x0101;
pub const VIRTIO_SND_R_PCM_PREPARE: u32 = 0x0102;
pub const VIRTIO_SND_R_PCM_RELEASE: u32 = 0x0103;
pub const VIRTIO_SND_R_PCM_START: u32 = 0x0104;
pub const VIRTIO_SND_R_PCM_STOP: u32 = 0x0105;
pub const VIRTIO_SND_R_CHMAP_INFO: u32 = 0x0200;

// Response status
pub const VIRTIO_SND_S_OK: u32 = 0x8000;
pub const VIRTIO_SND_S_BAD_MSG: u32 = 0x8001;
pub const VIRTIO_SND_S_NOT_SUPP: u32 = 0x8002;
pub const VIRTIO_SND_S_IO_ERR: u32 = 0x8003;

// Event types
pub const VIRTIO_SND_EVT_JACK_CONNECTED: u32 = 0x1000;
pub const VIRTIO_SND_EVT_JACK_DISCONNECTED: u32 = 0x1001;
pub const VIRTIO_SND_EVT_PCM_PERIOD_ELAPSED: u32 = 0x1100;
pub const VIRTIO_SND_EVT_PCM_XRUN: u32 = 0x1101;

// PCM Formats
pub const VIRTIO_SND_PCM_FMT_IMA_ADPCM: u8 = 0;
pub const VIRTIO_SND_PCM_FMT_MU_LAW: u8 = 1;
pub const VIRTIO_SND_PCM_FMT_A_LAW: u8 = 2;
pub const VIRTIO_SND_PCM_FMT_S8: u8 = 3;
pub const VIRTIO_SND_PCM_FMT_U8: u8 = 4;
pub const VIRTIO_SND_PCM_FMT_S16: u8 = 5;
pub const VIRTIO_SND_PCM_FMT_U16: u8 = 6;
pub const VIRTIO_SND_PCM_FMT_S18_3: u8 = 7;
pub const VIRTIO_SND_PCM_FMT_U18_3: u8 = 8;
pub const VIRTIO_SND_PCM_FMT_S20_3: u8 = 9;
pub const VIRTIO_SND_PCM_FMT_U20_3: u8 = 10;
pub const VIRTIO_SND_PCM_FMT_S24_3: u8 = 11;
pub const VIRTIO_SND_PCM_FMT_U24_3: u8 = 12;
pub const VIRTIO_SND_PCM_FMT_S20: u8 = 13;
pub const VIRTIO_SND_PCM_FMT_U20: u8 = 14;
pub const VIRTIO_SND_PCM_FMT_S24: u8 = 15;
pub const VIRTIO_SND_PCM_FMT_U24: u8 = 16;
pub const VIRTIO_SND_PCM_FMT_S32: u8 = 17;
pub const VIRTIO_SND_PCM_FMT_U32: u8 = 18;
pub const VIRTIO_SND_PCM_FMT_FLOAT: u8 = 19;
pub const VIRTIO_SND_PCM_FMT_FLOAT64: u8 = 20;

pub const VIRTIO_SND_PCM_RATE_5512: u8 = 0;
pub const VIRTIO_SND_PCM_RATE_8000: u8 = 1;
pub const VIRTIO_SND_PCM_RATE_11025: u8 = 2;
pub const VIRTIO_SND_PCM_RATE_16000: u8 = 3;
pub const VIRTIO_SND_PCM_RATE_22050: u8 = 4;
pub const VIRTIO_SND_PCM_RATE_32000: u8 = 5;
pub const VIRTIO_SND_PCM_RATE_44100: u8 = 6;
pub const VIRTIO_SND_PCM_RATE_48000: u8 = 7;
pub const VIRTIO_SND_PCM_RATE_64000: u8 = 8;
pub const VIRTIO_SND_PCM_RATE_88200: u8 = 9;
pub const VIRTIO_SND_PCM_RATE_96000: u8 = 10;
pub const VIRTIO_SND_PCM_RATE_176400: u8 = 11;
pub const VIRTIO_SND_PCM_RATE_192000: u8 = 12;
pub const VIRTIO_SND_PCM_RATE_384000: u8 = 13;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndConfig {
    pub jacks: u32,
    pub streams: u32,
    pub chmaps: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndHdr {
    pub code: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndEvent {
    pub hdr: VirtioSndHdr,
    pub data: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndQueryInfo {
    pub hdr: VirtioSndHdr,
    pub start_id: u32,
    pub count: u32,
    pub size: u32, // sizeof(info struct)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndPcmInfo {
    pub hdr: VirtioSndHdr, // h.code = VIRTIO_SND_R_PCM_INFO
    pub features: u32,
    pub formats: u64,
    pub rates: u64,
    pub direction: u8,
    pub channels_min: u8,
    pub channels_max: u8,
    pub padding: [u8; 5],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndPcmSetParams {
    pub hdr: VirtioSndHdr, // h.code = VIRTIO_SND_R_PCM_SET_PARAMS
    pub buffer_bytes: u32,
    pub period_bytes: u32,
    pub features: u32,
    pub channels: u8,
    pub format: u8,
    pub rate: u8,
    pub padding: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndPcmHdr {
    pub hdr: VirtioSndHdr, // h.code = VIRTIO_SND_R_PCM_PREPARE/START/STOP/RELEASE
    pub stream_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndPcmXfer {
    pub stream_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioSndPcmStatus {
    pub status: u32,
    pub latency_bytes: u32,
}
