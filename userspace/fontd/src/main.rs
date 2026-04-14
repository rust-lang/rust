#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::font_protocol::{
    decode_request_tag, EnsureGlyphs, EnsureGlyphsResp, FaceMetrics,
    FontRequestTag, FontResponseTag, GetFaceMetrics, GlyphPlacement,
};
use abi::ids::HandleId;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use ipc_helpers::rpc::RpcServer;
use petals::font::TextRenderer;
use stem::syscall::channel::channel_send_msg;
use stem::{error, info};

use petals::Atlas;

struct FontService {
    renderer: TextRenderer,
    // (face_id, px_size) -> Atlas
    atlases: BTreeMap<(u64, u16), Atlas>,
    // (face_id, px_size, glyph_id) -> GlyphPlacement
    cache: BTreeMap<(u64, u16, u32), GlyphPlacement>,
}

/// Unpack the paired channel handles from arg0.
///
/// sprout passes `(write_h << 16) | read_h` as the spawn argument so that
/// fontd has both ends of the request/reply channel pair.
fn unpack_handles(arg: usize) -> (u32, u32) {
    let write_h = ((arg >> 16) & 0xFFFF) as u32;
    let read_h = (arg & 0xFFFF) as u32;
    (write_h, read_h)
}

#[stem::main]
fn main(arg0: usize) -> ! {
    info!("fontd: starting up...");

    if arg0 == 0 {
        error!("fontd: No channel handles provided!");
        loop {
            stem::yield_now();
        }
    }

    let (write_h, read_h) = unpack_handles(arg0);
    info!("fontd: Entering RPC loop (read={}, write={})", read_h, write_h);

    let mut service = FontService {
        renderer: TextRenderer::load_from_boot("/share/fonts/NotoSans-Regular.ttf")
            .expect("Failed to load default font"),
        atlases: BTreeMap::new(),
        cache: BTreeMap::new(),
    };

    let mut server = RpcServer::new(read_h);

    loop {
        let req = match server.next() {
            Ok(r) => r,
            Err(_) => {
                stem::yield_now();
                continue;
            }
        };

        let tag = match decode_request_tag(&req.payload) {
            Some(tag) => tag,
            None => {
                error!("fontd: Received invalid request tag");
                let _ = server.reply_err(
                    req.request_id,
                    write_h,
                    abi::errors::Errno::EINVAL,
                );
                continue;
            }
        };

        match tag {
            FontRequestTag::Ping => {
                let resp = [FontResponseTag::Pong as u8];
                let _ = server.reply(req.request_id, write_h, &resp);
            }
            FontRequestTag::GetFaceMetrics => {
                if let Some(font_req) = GetFaceMetrics::decode(&req.payload[1..]) {
                    let font = &service.renderer.font;
                    let metrics = font
                        .horizontal_line_metrics(font_req.px_size as f32)
                        .unwrap_or_else(|| font.horizontal_line_metrics(16.0).unwrap());

                    let resp = FaceMetrics {
                        ascent: metrics.ascent as i16,
                        descent: metrics.descent as i16,
                        line_gap: metrics.line_gap as i16,
                        units_per_em: font.units_per_em() as u16,
                    };
                    let mut out_buf = [0u8; 10];
                    if let Some(len) = resp.encode(&mut out_buf) {
                        let _ = server.reply(req.request_id, write_h, &out_buf[..len]);
                    }
                } else {
                    let _ = server.reply_err(
                        req.request_id,
                        write_h,
                        abi::errors::Errno::EINVAL,
                    );
                }
            }
            FontRequestTag::EnsureGlyphs => {
                if let Some(font_req) = EnsureGlyphs::decode(&req.payload[1..]) {
                    handle_ensure_glyphs(
                        &mut service,
                        &server,
                        write_h,
                        req.request_id,
                        font_req,
                    );
                } else {
                    let _ = server.reply_err(
                        req.request_id,
                        write_h,
                        abi::errors::Errno::EINVAL,
                    );
                }
            }
        }
    }
}

fn handle_ensure_glyphs(
    service: &mut FontService,
    server: &RpcServer,
    write_h: u32,
    request_id: u64,
    req: EnsureGlyphs,
) {
    let face_id_u64 = req.face_id.to_u64_lossy();
    let px_size = req.px_size;

    // Get or create atlas
    if !service.atlases.contains_key(&(face_id_u64, px_size)) {
        let atlas = Atlas::new("font_atlas", 1024, 1024, 1).expect("Failed to create atlas");
        service.atlases.insert((face_id_u64, px_size), atlas);
    }
    let atlas = service.atlases.get_mut(&(face_id_u64, px_size)).unwrap();

    let mut placements = Vec::new();
    let mut missing = Vec::new();

    for &gid in &req.glyph_ids {
        if let Some(p) = service.cache.get(&(face_id_u64, px_size, gid)) {
            placements.push(*p);
            continue;
        }

        // Rasterize
        let (metrics, bitmap) = service
            .renderer
            .font
            .rasterize(char::from_u32(gid).unwrap_or(' '), px_size as f32);

        if metrics.width == 0 || metrics.height == 0 {
            let p = GlyphPlacement {
                glyph_id: gid,
                x: 0,
                y: 0,
                w: 0,
                h: 0,
                bearing_x: metrics.xmin as i16,
                bearing_y: metrics.ymin as i16,
                advance: metrics.advance_width as i16,
            };
            service.cache.insert((face_id_u64, px_size, gid), p);
            placements.push(p);
            continue;
        }

        // Pack
        if let Some((x, y)) = atlas.pack(metrics.width as u32, metrics.height as u32, &bitmap) {
            let p = GlyphPlacement {
                glyph_id: gid,
                x: x as u16,
                y: y as u16,
                w: metrics.width as u16,
                h: metrics.height as u16,
                bearing_x: metrics.xmin as i16,
                bearing_y: metrics.ymin as i16,
                advance: metrics.advance_width as i16,
            };
            service.cache.insert((face_id_u64, px_size, gid), p);
            placements.push(p);
        } else {
            missing.push(gid);
        }
    }

    let resp = EnsureGlyphsResp {
        req_face_id: req.face_id,
        req_px_size: px_size,
        atlas_fd: atlas.texture.fd,
        atlas_width: atlas.texture.width,
        atlas_height: atlas.texture.height,
        atlas_format: abi::font_protocol::AtlasFormat::A8,
        atlas_version: 1, // Need to increment this if we invalidate
        placements,
        missing,
    };

    let mut resp_buf = vec![0u8; 4096 * 4];
    if let Some(len) = resp.encode(&mut resp_buf) {
        // Send atlas fd alongside the encoded response framed with RpcHeader.
        let _ = channel_send_msg(write_h, &[], &[atlas.texture.fd]);
        let _ = server.reply(request_id, write_h, &resp_buf[..len]);
    }
}
