//! Intel HDA sound driver — VFS-first PCM audio.
//!
//! Exposes the HDA controller as a VFS provider mounted at `/dev/audio/card0/`:
//!
//! ```text
//! /dev/audio/card0/
//!     ctl     — control node (AUDIO_GET_INFO device call)
//!     out0    — PCM playback stream (write PCM, poll POLLOUT for space)
//! ```
//!
//! Apps open `out0`, send an `AUDIO_SET_PARAMS` device call, then `AUDIO_START`,
//! and stream PCM via `write()`.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::device::DeviceKind;
use abi::sound::{
    AudioParams, AudioSampleFormat, AudioState, AudioStatus, AudioStreamInfo, AUDIO_DRAIN,
    AUDIO_GET_INFO, AUDIO_GET_PARAMS, AUDIO_GET_STATUS, AUDIO_SET_PARAMS, AUDIO_START, AUDIO_STOP,
    format_bit,
};
use abi::vfs_rpc::{VfsRpcOp, VfsRpcReqHeader, VFS_RPC_MAX_REQ};
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;
use core::ptr::{read_volatile, write_volatile};
use stem::syscall::channel::{channel_create, channel_send_all, channel_try_recv};
use stem::syscall::vfs::vfs_mount;
use stem::syscall::{device_alloc_dma, device_claim, device_dma_phys, device_map_mmio};
use stem::{debug, error, info, warn};

const REG_GCAP: u32 = 0x00;
const REG_GCTL: u32 = 0x08;
const REG_STATESTS: u32 = 0x0e;
const REG_INTCTL: u32 = 0x20;
const REG_INTSTS: u32 = 0x24;

const REG_CORBLBASE: u32 = 0x40;
const REG_CORBUBASE: u32 = 0x44;
const REG_CORBWP: u32 = 0x48;
const REG_CORBRP: u32 = 0x4a;
const REG_CORBCTL: u32 = 0x4c;
const REG_CORBSTS: u32 = 0x4d;
const REG_CORBSIZE: u32 = 0x4e;

const REG_RIRBLBASE: u32 = 0x50;
const REG_RIRBUBASE: u32 = 0x54;
const REG_RIRBWP: u32 = 0x58;
const REG_RINTCNT: u32 = 0x5a;
const REG_RIRBCTL: u32 = 0x5c;
const REG_RIRBSTS: u32 = 0x5d;
const REG_RIRBSIZE: u32 = 0x5e;

const GCTL_CRST: u32 = 1;

const INTCTL_GIE: u32 = 1 << 31;
const INTCTL_CIE: u32 = 1 << 30;

const SD_CTL0: u32 = 0x00;
const SD_CTL2: u32 = 0x02;
const SD_LPIB: u32 = 0x04;
const SD_CBL: u32 = 0x08;
const SD_LVI: u32 = 0x0c;
const SD_FMT: u32 = 0x12;
const SD_BDPL: u32 = 0x18;
const SD_BDPU: u32 = 0x1c;

const SD_CTL_SRST: u8 = 1 << 0;
const SD_CTL_RUN: u8 = 1 << 1;
const SD_CTL_IOCE: u8 = 1 << 2;

const VERB_GET_PARAM: u16 = 0x0f00;
const VERB_SET_POWER_STATE: u16 = 0x0705;
const VERB_SET_CONV_STREAM_CHAN: u16 = 0x0706;
const VERB_SET_PIN_WIDGET_CTRL: u16 = 0x0707;
const VERB_SET_EAPD_BTL: u16 = 0x070c;
const VERB_SET_CONNECT_SEL: u16 = 0x0701;

const PARAM_NODE_COUNT: u8 = 0x04;
const PARAM_FG_TYPE: u8 = 0x05;
const PARAM_AWCAP: u8 = 0x09;
const PARAM_CONN_LEN: u8 = 0x0e;

const FG_TYPE_AUDIO: u8 = 0x01;
const WIDGET_AUDIO_OUT: u8 = 0x0;
const WIDGET_PIN: u8 = 0x4;

const CORB_ENTRIES: usize = 256;
const RIRB_ENTRIES: usize = 256;

const STREAM_TAG: u8 = 1;
const STREAM_CHAN: u8 = 0;
const STREAM_INDEX_MAX: usize = 31;

const BUFFER_BYTES: usize = 16 * 1024;
const BDL_ENTRY_COUNT: usize = 4;
const BDL_CHUNK_BYTES: usize = BUFFER_BYTES / BDL_ENTRY_COUNT;
const STREAM_FORMAT_48K_STEREO_16: u16 = 0x0011;

#[repr(C, align(128))]
#[derive(Copy, Clone, Default)]
struct BdlEntry {
    addr_lo: u32,
    addr_hi: u32,
    length: u32,
    ioc: u32,
}

struct HdaController {
    mmio: u64,

    corb_virt: u64,
    corb_phys: u64,
    rirb_virt: u64,
    rirb_phys: u64,
    corb_wp: u16,
    rirb_rp: u16,

    bdl_virt: u64,
    bdl_phys: u64,
    pcm_virt: u64,
    pcm_phys: u64,
    pcm_write_pos: usize,
    sd_base: u32,
    sd_index: usize,
}

#[stem::main]
fn main(boot_fd: usize) -> ! {
    debug!("HDAUDIO: starting (boot_fd={})", boot_fd);

    let mut path_buf = [0u8; 128];
    let path_len = if boot_fd != 0 {
        use stem::syscall::vfs::vfs_read;
        vfs_read(boot_fd as u32, &mut path_buf).unwrap_or(0)
    } else {
        0
    };

    let path_str = if path_len > 0 {
        core::str::from_utf8(&path_buf[..path_len])
            .unwrap_or("")
            .trim_matches(char::from(0))
    } else {
        ""
    };

    debug!("HDAUDIO: using device path: {}", path_str);

    let dev_path = if !path_str.is_empty() {
        alloc::string::String::from(path_str)
    } else {
        match find_hda_device() {
            Some(p) => p,
            None => {
                error!("HDAUDIO: no HDA PCI device found");
                loop {
                    stem::time::sleep_ms(1000);
                }
            }
        }
    };

    debug!("HDAUDIO: claiming PCI device at '{}'...", dev_path);
    let claim = match device_claim(&dev_path) {
        Ok(h) => h,
        Err(e) => {
            error!("HDAUDIO: claim failed for '{}': {:?}", dev_path, e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };
    debug!("HDAUDIO: mapping BAR0 MMIO for claim {}...", claim);
    let mmio = match device_map_mmio(claim, 0) {
        Ok(v) => v,
        Err(e) => {
            error!("HDAUDIO: BAR0 map failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    debug!("HDAUDIO: claimed device '{}' mmio=0x{:x}", dev_path, mmio);

    let mut hda = match HdaController::new(mmio, claim) {
        Ok(h) => h,
        Err(e) => {
            error!("HDAUDIO: init failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    if let Err(e) = hda.init_controller() {
        error!("HDAUDIO: controller init failed: {:?}", e);
        loop {
            stem::time::sleep_ms(1000);
        }
    }

    let codec_addr = match hda.first_codec_addr() {
        Some(c) => c,
        None => {
            error!("HDAUDIO: no codec present");
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };
    debug!("HDAUDIO: using codec address {}", codec_addr);

    let (afg, out_nid, pin_nid) = match hda.discover_audio_path(codec_addr) {
        Some(path) => path,
        None => {
            error!("HDAUDIO: failed to discover output path");
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };
    debug!(
        "HDAUDIO: path AFG=0x{:02x} OUT=0x{:02x} PIN=0x{:02x}",
        afg, out_nid, pin_nid
    );

    if let Err(e) = hda.configure_codec(codec_addr, afg, out_nid, pin_nid) {
        warn!("HDAUDIO: codec config failed: {:?}", e);
    }
    if let Err(e) = hda.setup_stream_dma() {
        error!("HDAUDIO: stream setup failed: {:?}", e);
        loop {
            stem::time::sleep_ms(1000);
        }
    }

    // ── Mount VFS provider at /dev/audio/card0/ ───────────────────────────────
    let (req_write, req_read) = match channel_create(VFS_RPC_MAX_REQ * 16) {
        Ok(h) => h,
        Err(e) => {
            error!("HDAUDIO: channel_create failed: {:?}", e);
            loop {
                stem::time::sleep_ms(1000);
            }
        }
    };

    match vfs_mount(req_write, "/dev/audio/card0") {
        Ok(()) => info!("HDAUDIO: Mounted at /dev/audio/card0"),
        Err(e) => warn!("HDAUDIO: vfs_mount failed: {:?}", e),
    }

    // ── Audio card state ──────────────────────────────────────────────────────
    let mut card = HdaAudioCard::new();
    let mut rpc_buf = vec![0u8; VFS_RPC_MAX_REQ];
    let hdr_size = size_of::<VfsRpcReqHeader>();
    let mut total_bytes: u64 = 0;
    let mut last_log_ns = 0u64;

    // ── Main event loop ───────────────────────────────────────────────────────
    loop {
        // 1. Service VFS RPC requests (non-blocking).
        loop {
            match channel_try_recv(req_read, &mut rpc_buf) {
                Ok(n) if n >= hdr_size => {
                    let hdr: VfsRpcReqHeader = unsafe {
                        core::ptr::read_unaligned(rpc_buf.as_ptr() as *const VfsRpcReqHeader)
                    };
                    let op = match VfsRpcOp::from_u8(hdr.op) {
                        Some(o) => o,
                        None => {
                            let _ = channel_send_all(hdr.resp_port, &hda_resp_err(22));
                            continue;
                        }
                    };
                    let payload = &rpc_buf[hdr_size..n];
                    let resp = hda_dispatch_rpc(op, payload, &mut card);
                    let _ = channel_send_all(hdr.resp_port, &resp);

                    // Notify subscribers when ring gains space.
                    if card.out0_subscribed && card.ring.free_space() > 0 {
                        let _ = stem::syscall::vfs::vfs_notify(
                            req_write,
                            HANDLE_OUT0,
                            abi::syscall::poll_flags::POLLOUT,
                        );
                    }
                }
                _ => break,
            }
        }

        // 2. Feed HDA hardware from ring buffer.
        let chunk = 4096usize;
        if card.ring.available() >= chunk {
            let mut tmp = vec![0u8; chunk];
            let n = card.ring.dequeue(&mut tmp);
            if n > 0 {
                hda.feed_pcm(&tmp[..n]);
                total_bytes = total_bytes.saturating_add(n as u64);
                let bpf = {
                    let fmt = AudioSampleFormat::from_u32(card.params.sample_format)
                        .unwrap_or(AudioSampleFormat::S16LE);
                    (fmt.bytes_per_sample() * card.params.channels) as usize
                };
                card.app_frame += (n / bpf.max(1)) as u64;
                let now = stem::time::monotonic_ns();
                if now.saturating_sub(last_log_ns) > 1_000_000_000 {
                    debug!("HDAUDIO: streamed {} bytes", total_bytes);
                    last_log_ns = now;
                }
                // Notify waiting writers.
                if card.out0_subscribed {
                    let _ = stem::syscall::vfs::vfs_notify(
                        req_write,
                        HANDLE_OUT0,
                        abi::syscall::poll_flags::POLLOUT,
                    );
                }
            }
        } else {
            stem::time::sleep_ms(1);
        }
    }
}

// ── VFS provider handle constants ─────────────────────────────────────────────

const HANDLE_ROOT: u64 = 0;
const HANDLE_CTL: u64 = 1;
const HANDLE_OUT0: u64 = 2;

const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;

// ── HDA ring buffer ───────────────────────────────────────────────────────────

struct HdaRingBuf {
    data: Vec<u8>,
    head: usize,
    tail: usize,
    len: usize,
    cap: usize,
}

impl HdaRingBuf {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let mut data = Vec::with_capacity(cap);
        data.resize(cap, 0u8);
        Self { data, head: 0, tail: 0, len: 0, cap }
    }
    fn free_space(&self) -> usize { self.cap - self.len }
    fn available(&self) -> usize { self.len }
    fn enqueue(&mut self, src: &[u8]) -> usize {
        let n = src.len().min(self.free_space());
        for i in 0..n {
            self.data[self.tail] = src[i];
            self.tail = (self.tail + 1) % self.cap;
        }
        self.len += n;
        n
    }
    fn dequeue(&mut self, dst: &mut [u8]) -> usize {
        let n = dst.len().min(self.len);
        for i in 0..n {
            dst[i] = self.data[self.head];
            self.head = (self.head + 1) % self.cap;
        }
        self.len -= n;
        n
    }
}

// ── HDA audio card state ──────────────────────────────────────────────────────

struct HdaAudioCard {
    ring: HdaRingBuf,
    params: AudioParams,
    state: u32,
    app_frame: u64,
    xruns: u32,
    out0_subscribed: bool,
}

impl HdaAudioCard {
    fn new() -> Self {
        Self {
            ring: HdaRingBuf::new(64 * 1024),
            params: AudioParams {
                sample_format: AudioSampleFormat::S16LE as u32,
                rate: 48000,
                channels: 2,
                period_frames: 1024,
                buffer_frames: 4096,
                _reserved: [0; 3],
            },
            state: AudioState::Stopped as u32,
            app_frame: 0,
            xruns: 0,
            out0_subscribed: false,
        }
    }

    fn stream_info(&self) -> AudioStreamInfo {
        AudioStreamInfo {
            supported_formats: format_bit(AudioSampleFormat::S16LE),
            min_rate: 44100,
            max_rate: 48000,
            max_channels: 2,
            min_buffer_frames: 256,
            max_buffer_frames: 65536,
            min_period_frames: 64,
            current_params: self.params,
            _reserved: [0; 4],
        }
    }

    fn status(&self) -> AudioStatus {
        let fmt = AudioSampleFormat::from_u32(self.params.sample_format)
            .unwrap_or(AudioSampleFormat::S16LE);
        let bpf = (fmt.bytes_per_sample() * self.params.channels) as usize;
        let avail = if bpf > 0 { (self.ring.free_space() / bpf) as u32 } else { 0 };
        AudioStatus {
            state: self.state,
            hw_frame: self.app_frame,
            app_frame: self.app_frame,
            avail_frames: avail,
            xruns: self.xruns,
            _reserved: [0; 4],
        }
    }
}

// ── VFS RPC helpers ───────────────────────────────────────────────────────────

fn hda_resp_ok_u64(v: u64) -> Vec<u8> {
    let mut b = vec![0u8; 9];
    b[1..9].copy_from_slice(&v.to_le_bytes());
    b
}

fn hda_resp_ok_stat(mode: u32, size: u64, ino: u64) -> Vec<u8> {
    let mut b = vec![0u8; 21];
    b[1..5].copy_from_slice(&mode.to_le_bytes());
    b[5..13].copy_from_slice(&size.to_le_bytes());
    b[13..21].copy_from_slice(&ino.to_le_bytes());
    b
}

fn hda_resp_ok_read(data: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; 5 + data.len()];
    b[1..5].copy_from_slice(&(data.len() as u32).to_le_bytes());
    b[5..].copy_from_slice(data);
    b
}

fn hda_resp_ok_written(n: u32) -> Vec<u8> {
    let mut b = vec![0u8; 5];
    b[1..5].copy_from_slice(&n.to_le_bytes());
    b
}

fn hda_resp_ok_poll(revents: u32) -> Vec<u8> {
    let mut b = vec![0u8; 5];
    b[1..5].copy_from_slice(&revents.to_le_bytes());
    b
}

fn hda_resp_ok_dc(ret: u32, out: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; 9 + out.len()];
    b[1..5].copy_from_slice(&ret.to_le_bytes());
    b[5..9].copy_from_slice(&(out.len() as u32).to_le_bytes());
    b[9..].copy_from_slice(out);
    b
}

fn hda_resp_err(e: u8) -> Vec<u8> { vec![e] }

fn hda_encode_readdir(names: &[(&str, u32, u64)], offset: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, (name, ftype, ino)) in names.iter().enumerate() {
        if (i as u64) < offset { continue; }
        let nb = name.as_bytes();
        let nl = nb.len().min(255) as u8;
        out.extend_from_slice(&ino.to_le_bytes());
        out.push(*ftype as u8);
        out.push(nl);
        out.extend_from_slice(&nb[..nl as usize]);
    }
    out
}

fn hda_dispatch_rpc(op: VfsRpcOp, payload: &[u8], card: &mut HdaAudioCard) -> Vec<u8> {
    match op {
        VfsRpcOp::Lookup => {
            if payload.len() < 4 { return hda_resp_err(22); }
            let plen = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
            let path = if payload.len() >= 4 + plen {
                core::str::from_utf8(&payload[4..4 + plen]).unwrap_or("")
            } else { "" };
            let h = match path { "" => HANDLE_ROOT, "ctl" => HANDLE_CTL, "out0" => HANDLE_OUT0, _ => return hda_resp_err(2) };
            hda_resp_ok_u64(h)
        }
        VfsRpcOp::Stat => {
            if payload.len() < 8 { return hda_resp_err(22); }
            let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
            match h {
                HANDLE_ROOT => hda_resp_ok_stat(S_IFDIR | 0o755, 0, 1),
                HANDLE_CTL  => hda_resp_ok_stat(S_IFREG | 0o444, 0, 2),
                HANDLE_OUT0 => hda_resp_ok_stat(S_IFREG | 0o222, 0, 3),
                _ => hda_resp_err(2),
            }
        }
        VfsRpcOp::Readdir => {
            if payload.len() < 20 { return hda_resp_err(22); }
            let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
            let off = u64::from_le_bytes(payload[8..16].try_into().unwrap_or([0;8]));
            if h != HANDLE_ROOT { return hda_resp_err(20); }
            let entries: &[(&str, u32, u64)] = &[("ctl", S_IFREG, 2), ("out0", S_IFREG, 3)];
            hda_resp_ok_read(&hda_encode_readdir(entries, off))
        }
        VfsRpcOp::Read => hda_resp_ok_read(&[]),
        VfsRpcOp::Write => {
            if payload.len() < 20 { return hda_resp_err(22); }
            let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
            if h != HANDLE_OUT0 { return hda_resp_err(30); }
            let dlen = u32::from_le_bytes(payload[16..20].try_into().unwrap_or([0;4])) as usize;
            let data = if payload.len() >= 20 + dlen { &payload[20..20+dlen] } else { &payload[20..] };
            let n = card.ring.enqueue(data);
            hda_resp_ok_written(n as u32)
        }
        VfsRpcOp::Poll => {
            if payload.len() < 8 { return hda_resp_err(22); }
            let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
            if h != HANDLE_OUT0 { return hda_resp_ok_poll(abi::syscall::poll_flags::POLLOUT as u32); }
            let rev = if card.ring.free_space() > 0 { abi::syscall::poll_flags::POLLOUT as u32 } else { 0 };
            hda_resp_ok_poll(rev)
        }
        VfsRpcOp::DeviceCall => {
            let dc_size = size_of::<abi::device::DeviceCall>();
            if payload.len() < 8 + dc_size { return hda_resp_err(22); }
            let dc: abi::device::DeviceCall = unsafe {
                core::ptr::read_unaligned(payload[8..].as_ptr() as *const abi::device::DeviceCall)
            };
            if dc.kind != DeviceKind::Audio { return hda_resp_err(38); }
            let in_data = &payload[8 + dc_size..];
            match dc.op {
                AUDIO_GET_INFO => {
                    let info = card.stream_info();
                    let bytes = unsafe { core::slice::from_raw_parts(&info as *const AudioStreamInfo as *const u8, size_of::<AudioStreamInfo>()) };
                    hda_resp_ok_dc(0, bytes)
                }
                AUDIO_SET_PARAMS => {
                    if in_data.len() >= size_of::<AudioParams>() {
                        let req: AudioParams = unsafe { core::ptr::read_unaligned(in_data.as_ptr() as *const AudioParams) };
                        card.params = AudioParams { sample_format: req.sample_format, rate: req.rate.clamp(44100, 48000), channels: req.channels.clamp(1,2), period_frames: req.period_frames.max(64), buffer_frames: req.buffer_frames.max(256), _reserved: [0;3] };
                    }
                    let bytes = unsafe { core::slice::from_raw_parts(&card.params as *const AudioParams as *const u8, size_of::<AudioParams>()) };
                    hda_resp_ok_dc(0, bytes)
                }
                AUDIO_GET_PARAMS => {
                    let bytes = unsafe { core::slice::from_raw_parts(&card.params as *const AudioParams as *const u8, size_of::<AudioParams>()) };
                    hda_resp_ok_dc(0, bytes)
                }
                AUDIO_GET_STATUS => {
                    let st = card.status();
                    let bytes = unsafe { core::slice::from_raw_parts(&st as *const AudioStatus as *const u8, size_of::<AudioStatus>()) };
                    hda_resp_ok_dc(0, bytes)
                }
                AUDIO_START   => { card.state = AudioState::Running as u32;  hda_resp_ok_dc(0, &[]) }
                AUDIO_STOP    => { card.state = AudioState::Stopped as u32; card.ring = HdaRingBuf::new(64*1024); hda_resp_ok_dc(0, &[]) }
                AUDIO_DRAIN   => { card.state = AudioState::Draining as u32; hda_resp_ok_dc(0, &[]) }
                _             => hda_resp_err(38),
            }
        }
        VfsRpcOp::SubscribeReady => {
            if payload.len() >= 8 {
                let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
                if h == HANDLE_OUT0 { card.out0_subscribed = true; }
            }
            vec![0u8]
        }
        VfsRpcOp::UnsubscribeReady => {
            if payload.len() >= 8 {
                let h = u64::from_le_bytes(payload[..8].try_into().unwrap_or([0;8]));
                if h == HANDLE_OUT0 { card.out0_subscribed = false; }
            }
            vec![0u8]
        }
        VfsRpcOp::Close  => vec![0u8],
        VfsRpcOp::Rename => hda_resp_err(30),
    }
}

impl HdaController {
    fn new(mmio: u64, claim: usize) -> Result<Self, abi::errors::Errno> {
        let corb_virt = device_alloc_dma(claim, 1)?;
        let corb_phys = device_dma_phys(corb_virt)?;
        let rirb_virt = device_alloc_dma(claim, 1)?;
        let rirb_phys = device_dma_phys(rirb_virt)?;

        let bdl_virt = device_alloc_dma(claim, 1)?;
        let bdl_phys = device_dma_phys(bdl_virt)?;
        let pcm_virt = device_alloc_dma(claim, BUFFER_BYTES / 4096)?;
        let pcm_phys = device_dma_phys(pcm_virt)?;

        Ok(Self {
            mmio,
            corb_virt,
            corb_phys,
            rirb_virt,
            rirb_phys,
            corb_wp: 0,
            rirb_rp: 0,
            bdl_virt,
            bdl_phys,
            pcm_virt,
            pcm_phys,
            pcm_write_pos: 0,
            sd_base: 0,
            sd_index: 0,
        })
    }

    fn init_controller(&mut self) -> Result<(), abi::errors::Errno> {
        self.reset_controller();
        stem::sleep_ms(10);
        self.init_corb_rirb();
        let gcap = self.read_u16(REG_GCAP);
        let state_sts = self.read_u16(REG_STATESTS);
        info!(
            "HDAUDIO: GCAP=0x{:04x} STATESTS=0x{:04x} codecs_mask=0x{:x}",
            gcap,
            state_sts,
            state_sts & 0x7
        );

        let iss = ((gcap >> 8) & 0x0f) as usize;
        let oss = ((gcap >> 12) & 0x0f) as usize;
        if oss == 0 {
            return Err(abi::errors::Errno::ENODEV);
        }
        self.sd_index = iss;
        if self.sd_index > STREAM_INDEX_MAX {
            return Err(abi::errors::Errno::EINVAL);
        }
        self.sd_base = 0x80 + (self.sd_index as u32) * 0x20;

        let stream_irq_bit = 1u32 << self.sd_index;
        self.write_u32(REG_INTCTL, INTCTL_GIE | INTCTL_CIE | stream_irq_bit);
        self.write_u32(REG_INTSTS, 0xffff_ffff);
        Ok(())
    }

    fn first_codec_addr(&self) -> Option<u8> {
        let sts = self.read_u16(REG_STATESTS);
        for cad in 0..=14u8 {
            if (sts & (1u16 << cad)) != 0 {
                return Some(cad);
            }
        }
        None
    }

    fn discover_audio_path(&mut self, cad: u8) -> Option<(u8, u8, u8)> {
        let root_nc = self.get_param(cad, 0x00, PARAM_NODE_COUNT)?;
        let root_start = ((root_nc >> 16) & 0x7f) as u8;
        let root_count = (root_nc & 0x7f) as u8;
        stem::debug!(
            "HDAUDIO: root nodes: start={} count={}",
            root_start,
            root_count
        );

        let mut afg = None;
        for nid in root_start..root_start.saturating_add(root_count) {
            let fg = self.get_param(cad, nid, PARAM_FG_TYPE)?;
            stem::debug!("HDAUDIO: node {} fg_type=0x{:08x}", nid, fg);
            if (fg & 0xff) as u8 == FG_TYPE_AUDIO {
                afg = Some(nid);
                break;
            }
        }
        let afg = match afg {
            Some(a) => a,
            None => {
                error!("HDAUDIO: no Audio Function Group found");
                return None;
            }
        };
        stem::debug!("HDAUDIO: AFG at node {}", afg);

        let sub = self.get_param(cad, afg, PARAM_NODE_COUNT)?;
        let start = ((sub >> 16) & 0x7f) as u8;
        let count = (sub & 0x7f) as u8;
        stem::debug!("HDAUDIO: AFG sub-nodes: start={} count={}", start, count);

        let mut out_nid = None;
        let mut pin_nid = None;
        for nid in start..start.saturating_add(count) {
            let awcap = self.get_param(cad, nid, PARAM_AWCAP)?;
            let wtype = ((awcap >> 20) & 0xf) as u8;
            let wtype_name = match wtype {
                0 => "AudioOut",
                1 => "AudioIn",
                2 => "AudioMix",
                3 => "AudioSel",
                4 => "PinComplex",
                5 => "Power",
                6 => "VolumeKnob",
                7 => "BeepGen",
                0xf => "VendorDefined",
                _ => "Unknown",
            };
            stem::debug!(
                "HDAUDIO:   widget nid={} type={}({}) awcap=0x{:08x}",
                nid,
                wtype,
                wtype_name,
                awcap
            );
            if wtype == WIDGET_AUDIO_OUT && out_nid.is_none() {
                out_nid = Some(nid);
            }
            if wtype == WIDGET_PIN && pin_nid.is_none() {
                pin_nid = Some(nid);
            }
            if out_nid.is_some() && pin_nid.is_some() {
                break;
            }
        }

        if out_nid.is_none() {
            error!("HDAUDIO: no AudioOut widget found (codec may be input-only, e.g. hda-micro)");
        }
        if pin_nid.is_none() {
            error!("HDAUDIO: no PinComplex widget found");
        }

        Some((afg, out_nid?, pin_nid?))
    }

    fn configure_codec(
        &mut self,
        cad: u8,
        afg: u8,
        out_nid: u8,
        pin_nid: u8,
    ) -> Result<(), abi::errors::Errno> {
        let _ = self.verb(cad, afg, VERB_SET_POWER_STATE, 0x00)?;
        let _ = self.verb(cad, out_nid, VERB_SET_POWER_STATE, 0x00)?;
        let _ = self.verb(cad, pin_nid, VERB_SET_POWER_STATE, 0x00)?;

        if let Some(conn_len) = self.get_param(cad, pin_nid, PARAM_CONN_LEN) {
            if (conn_len & 0x7f) != 0 {
                let _ = self.verb(cad, pin_nid, VERB_SET_CONNECT_SEL, 0x00)?;
            }
        }

        let _ = self.verb(cad, pin_nid, VERB_SET_PIN_WIDGET_CTRL, 0x40)?;
        let _ = self.verb(cad, pin_nid, VERB_SET_EAPD_BTL, 0x02)?;

        let sc = ((STREAM_TAG & 0x0f) << 4) | (STREAM_CHAN & 0x0f);
        let _ = self.verb(cad, out_nid, VERB_SET_CONV_STREAM_CHAN, sc)?;
        Ok(())
    }

    fn setup_stream_dma(&mut self) -> Result<(), abi::errors::Errno> {
        let pcm_phys = self.pcm_phys;
        let entries = self.bdl_ptr();
        for (i, ent) in entries.iter_mut().enumerate().take(BDL_ENTRY_COUNT) {
            let addr = pcm_phys + (i as u64) * (BDL_CHUNK_BYTES as u64);
            ent.addr_lo = addr as u32;
            ent.addr_hi = (addr >> 32) as u32;
            ent.length = BDL_CHUNK_BYTES as u32;
            ent.ioc = 1;
        }

        self.write_u8(self.sd_base + SD_CTL0, 0);
        self.write_u8(self.sd_base + SD_CTL0, SD_CTL_SRST);
        let start = stem::time::monotonic_ns();
        loop {
            if (self.read_u8(self.sd_base + SD_CTL0) & SD_CTL_SRST) != 0 {
                break;
            }
            if stem::time::monotonic_ns().saturating_sub(start) > 500_000_000 {
                break;
            }
            stem::sleep_ms(1);
        }
        self.write_u8(self.sd_base + SD_CTL0, 0);
        let start2 = stem::time::monotonic_ns();
        loop {
            if (self.read_u8(self.sd_base + SD_CTL0) & SD_CTL_SRST) == 0 {
                break;
            }
            if stem::time::monotonic_ns().saturating_sub(start2) > 500_000_000 {
                break;
            }
            stem::sleep_ms(1);
        }

        self.write_u32(self.sd_base + SD_BDPL, self.bdl_phys as u32);
        self.write_u32(self.sd_base + SD_BDPU, (self.bdl_phys >> 32) as u32);
        self.write_u32(self.sd_base + SD_CBL, BUFFER_BYTES as u32);
        self.write_u16(self.sd_base + SD_LVI, (BDL_ENTRY_COUNT - 1) as u16);
        self.write_u16(self.sd_base + SD_FMT, STREAM_FORMAT_48K_STEREO_16);

        let ctl2 = (STREAM_TAG & 0x0f) << 4;
        self.write_u8(self.sd_base + SD_CTL2, ctl2);
        self.write_u8(self.sd_base + SD_CTL0, SD_CTL_IOCE | SD_CTL_RUN);
        Ok(())
    }

    fn feed_pcm(&mut self, pcm: &[u8]) {
        if pcm.is_empty() {
            return;
        }

        let mut remain = pcm;
        while !remain.is_empty() {
            let to_end = BUFFER_BYTES - self.pcm_write_pos;
            let chunk = core::cmp::min(to_end, remain.len());
            unsafe {
                core::ptr::copy_nonoverlapping(
                    remain.as_ptr(),
                    (self.pcm_virt as *mut u8).add(self.pcm_write_pos),
                    chunk,
                );
            }
            self.pcm_write_pos = (self.pcm_write_pos + chunk) % BUFFER_BYTES;
            remain = &remain[chunk..];
        }
    }

    fn buffered_bytes(&self) -> usize {
        let lpib = self.read_u32(self.sd_base + SD_LPIB) as usize;
        if self.pcm_write_pos >= lpib {
            self.pcm_write_pos - lpib
        } else {
            BUFFER_BYTES - lpib + self.pcm_write_pos
        }
    }

    fn get_param(&mut self, cad: u8, nid: u8, param: u8) -> Option<u32> {
        self.verb(cad, nid, VERB_GET_PARAM, param).ok()
    }

    fn verb(
        &mut self,
        cad: u8,
        nid: u8,
        verb: u16,
        payload: u8,
    ) -> Result<u32, abi::errors::Errno> {
        let cmd =
            ((cad as u32) << 28) | ((nid as u32) << 20) | ((verb as u32) << 8) | (payload as u32);

        let next_wp = ((self.corb_wp as usize + 1) % CORB_ENTRIES) as u16;
        unsafe {
            write_volatile((self.corb_virt as *mut u32).add(next_wp as usize), cmd);
        }
        self.corb_wp = next_wp;
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        self.write_u16(REG_CORBWP, self.corb_wp);

        let start = stem::time::monotonic_ns();
        let timeout_ns = 500_000_000;

        loop {
            let wp = self.read_u16(REG_RIRBWP) & 0x00ff;
            if wp != self.rirb_rp {
                let prev_rp = self.rirb_rp;
                self.rirb_rp = wp;

                let mut last_resp = 0;
                for i in (prev_rp + 1)..=wp {
                    let idx = (i as usize) % RIRB_ENTRIES;
                    last_resp = unsafe {
                        core::ptr::read_volatile((self.rirb_virt as *const u32).add(idx * 2))
                    };
                }
                return Ok(last_resp);
            }

            if stem::time::monotonic_ns().saturating_sub(start) > timeout_ns {
                break;
            }
            stem::sleep_ms(1);
        }
        Err(abi::errors::Errno::ETIMEDOUT)
    }

    fn init_corb_rirb(&mut self) {
        self.write_u8(REG_CORBCTL, 0);
        self.write_u8(REG_RIRBCTL, 0);

        self.write_u8(REG_CORBSTS, 0xff);
        self.write_u8(REG_RIRBSTS, 0xff);

        let csz = self.read_u8(REG_CORBSIZE) & 0xf0;
        self.write_u8(REG_CORBSIZE, csz | 0x02);
        let rsz = self.read_u8(REG_RIRBSIZE) & 0xf0;
        self.write_u8(REG_RIRBSIZE, rsz | 0x02);

        self.write_u32(REG_CORBLBASE, self.corb_phys as u32);
        self.write_u32(REG_CORBUBASE, (self.corb_phys >> 32) as u32);
        self.write_u32(REG_RIRBLBASE, self.rirb_phys as u32);
        self.write_u32(REG_RIRBUBASE, (self.rirb_phys >> 32) as u32);

        self.write_u16(REG_CORBRP, 1 << 15);
        self.write_u16(REG_CORBWP, 0);
        self.corb_wp = 0;

        self.write_u16(REG_RIRBWP, 1 << 15);
        self.write_u16(REG_RINTCNT, 1);
        self.rirb_rp = self.read_u16(REG_RIRBWP) & 0x00ff;

        self.write_u8(REG_CORBCTL, 0x02);
        self.write_u8(REG_RIRBCTL, 0x03);
    }

    fn reset_controller(&self) {
        let mut gctl = self.read_u32(REG_GCTL);
        gctl &= !GCTL_CRST;
        self.write_u32(REG_GCTL, gctl);

        let start = stem::time::monotonic_ns();
        loop {
            if (self.read_u32(REG_GCTL) & GCTL_CRST) == 0 {
                break;
            }
            if stem::time::monotonic_ns().saturating_sub(start) > 500_000_000 {
                break;
            }
            stem::sleep_ms(1);
        }

        gctl |= GCTL_CRST;
        self.write_u32(REG_GCTL, gctl);

        let start2 = stem::time::monotonic_ns();
        loop {
            if (self.read_u32(REG_GCTL) & GCTL_CRST) != 0 {
                return;
            }
            if stem::time::monotonic_ns().saturating_sub(start2) > 500_000_000 {
                break;
            }
            stem::sleep_ms(1);
        }
        warn!("HDAUDIO: controller reset timeout");
    }

    fn bdl_ptr(&mut self) -> &mut [BdlEntry; BDL_ENTRY_COUNT] {
        unsafe { &mut *(self.bdl_virt as *mut [BdlEntry; BDL_ENTRY_COUNT]) }
    }

    fn read_u8(&self, reg: u32) -> u8 {
        unsafe { read_volatile((self.mmio + reg as u64) as *const u8) }
    }
    fn read_u16(&self, reg: u32) -> u16 {
        unsafe { read_volatile((self.mmio + reg as u64) as *const u16) }
    }
    fn read_u32(&self, reg: u32) -> u32 {
        unsafe { read_volatile((self.mmio + reg as u64) as *const u32) }
    }
    fn write_u8(&self, reg: u32, val: u8) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u8, val) }
    }
    fn write_u16(&self, reg: u32, val: u16) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u16, val) }
    }
    fn write_u32(&self, reg: u32, val: u32) {
        unsafe { write_volatile((self.mmio + reg as u64) as *mut u32, val) }
    }
}

/// Scan `/sys/devices` for an HDA controller (PCI class 0x0403) and return its sysfs path.
fn find_hda_device() -> Option<alloc::string::String> {
    use abi::syscall::vfs_flags;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_readdir};

    stem::debug!("HDAUDIO: Searching for HDA controller in /sys/devices...");
    let fd = match vfs_open("/sys/devices", vfs_flags::O_RDONLY) {
        Ok(fd) => fd,
        Err(e) => {
            stem::error!("HDAUDIO: Failed to open /sys/devices: {:?}", e);
            return None;
        }
    };

    let mut buf = [0u8; 4096];
    let n = match vfs_readdir(fd, &mut buf) {
        Ok(n) => n,
        Err(e) => {
            stem::error!("HDAUDIO: readdir(/sys/devices) failed: {:?}", e);
            let _ = vfs_close(fd);
            return None;
        }
    };
    let _ = vfs_close(fd);

    let mut pos = 0;
    while pos < n {
        let entry_buf = &buf[pos..n];
        let name = core::str::from_utf8(entry_buf)
            .unwrap_or("")
            .split('\0')
            .next()
            .unwrap_or("");
        if name.is_empty() {
            break;
        }

        if name.starts_with("pci-") {
            let class_path = alloc::format!("/sys/devices/{}/class", name);
            let class_str = read_sys_string(&class_path).unwrap_or("".to_string());

            stem::debug!(
                "HDAUDIO: Checking device {} class={}",
                name,
                class_str.trim()
            );

            if class_str.trim().starts_with("0x0403") {
                let dev_path = alloc::format!("/sys/devices/{}", name);
                stem::debug!("HDAUDIO: Found HDA controller via scan: {}", dev_path);
                return Some(dev_path);
            }
        }
        pos += name.len() + 1;
    }
    stem::warn!("HDAUDIO: No HDA controller found in /sys/devices.");
    None
}

fn read_sys_string(path: &str) -> Option<alloc::string::String> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY).ok()?;
    let mut buf = [0u8; 128];
    let n = vfs_read(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);

    Some(
        alloc::string::String::from_utf8_lossy(&buf[..n])
            .trim()
            .to_string(),
    )
}
