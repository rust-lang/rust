#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::blit::PixelBuffer;
use crate::geometry::{FillRule, Path2D, PathVerb, PointF, Rect, Transform2D};
use alloc::vec;
use alloc::vec::Vec;

const SUPERSAMPLE_SCALE: i32 = 4;

#[derive(Clone, Copy)]
struct Edge {
    x: i32,
    dx_dy: i32,
    y_min: i32,
    y_max: i32,
    winding: i32,
}

fn fixed_floor(val: i32) -> i32 {
    val / SUPERSAMPLE_SCALE
}

pub fn fill_rect_copy(dst: &mut PixelBuffer, x: i32, y: i32, w: i32, h: i32, color: u32) {
    let clip = Rect::new(0, 0, dst.width as i32, dst.height as i32);
    let r = Rect::new(x, y, w, h).clip(clip);
    if r.is_empty() {
        return;
    }

    for row_y in r.y()..r.y() + r.height() {
        let row = dst.row_mut(row_y as u32);
        row[r.x() as usize..(r.x() + r.width()) as usize].fill(color);
    }
}

pub fn fill_path(
    surface: &mut PixelBuffer,
    path: &Path2D,
    transform: &Transform2D,
    color: u32,
    fill_rule: FillRule,
    clip: &Rect,
) {
    let sa = ((color >> 24) & 0xFF) as u8;
    if sa == 0 {
        return;
    }

    // Simplification: use a coverage buffer for AA
    let mut edges = build_edges(path, transform, SUPERSAMPLE_SCALE);
    edges.sort_by(|a, b| a.y_min.cmp(&b.y_min));

    let clip_w = clip.width();
    let clip_h = clip.height();
    if clip_w <= 0 || clip_h <= 0 {
        return;
    }

    let mut coverage = vec![0u8; (clip_w as usize) * (clip_h as usize)];
    let y_limit = (clip.y() + clip.height()) * SUPERSAMPLE_SCALE;

    let mut active_edges: Vec<Edge> = Vec::with_capacity(16);
    let mut edge_idx = 0;

    for y_sub in (clip.y() * SUPERSAMPLE_SCALE)..y_limit {
        while edge_idx < edges.len() && edges[edge_idx].y_min <= y_sub {
            if edges[edge_idx].y_max > y_sub {
                active_edges.push(edges[edge_idx]);
            }
            edge_idx += 1;
        }

        active_edges.retain(|e| e.y_max > y_sub);
        if active_edges.is_empty() {
            continue;
        }

        active_edges.sort_by(|a, b| a.x.cmp(&b.x));

        let mut winding = 0;
        let mut start_x = 0;
        for i in 0..active_edges.len() {
            let x = fixed_floor(active_edges[i].x);
            if winding == 0 {
                start_x = x;
            }

            match fill_rule {
                FillRule::EvenOdd => winding ^= 1,
                FillRule::NonZero => winding += active_edges[i].winding,
            }

            if winding == 0 {
                let end_x = x;
                let start = start_x.max(clip.x()).min(clip.x() + clip.width());
                let end = end_x.max(clip.x()).min(clip.x() + clip.width());
                for xx in start..end {
                    let ix = (xx - clip.x()) as usize;
                    let iy = (y_sub / SUPERSAMPLE_SCALE - clip.y()) as usize;
                    coverage[iy * (clip_w as usize) + ix] =
                        coverage[iy * (clip_w as usize) + ix].saturating_add(1);
                }
            }
        }

        // Update x for active edges
        for e in &mut active_edges {
            e.x += e.dx_dy;
        }
    }

    // Blend coverage with surface
    let sr = ((color >> 16) & 0xFF) as u32;
    let sg = ((color >> 8) & 0xFF) as u32;
    let sb = (color & 0xFF) as u32;
    let max_coverage = (SUPERSAMPLE_SCALE * SUPERSAMPLE_SCALE) as u32;

    for y in 0..clip_h {
        let row = surface.row_mut((clip.y() + y) as u32);
        for x in 0..clip_w {
            let cov = coverage[(y * clip_w + x) as usize] as u32;
            if cov == 0 {
                continue;
            }

            let alpha = (sa as u32 * cov) / max_coverage;
            let dst_idx = (clip.x() + x) as usize;
            let dst_color = row[dst_idx];

            row[dst_idx] = blend(dst_color, (alpha << 24) | (sr << 16) | (sg << 8) | sb);
        }
    }
}

fn build_edges(path: &Path2D, transform: &Transform2D, scale: i32) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut last_p = (0.0f32, 0.0f32);
    let mut start_p = (0.0f32, 0.0f32);

    for verb in &path.verbs {
        match verb {
            PathVerb::MoveTo(p) => {
                last_p = (p.x, p.y);
                start_p = last_p;
            }
            PathVerb::LineTo(p) => {
                add_edge(&mut edges, last_p, (p.x, p.y), transform, scale);
                last_p = (p.x, p.y);
            }
            PathVerb::QuadTo(cp, p) => {
                let mut points = Vec::new();
                crate::tessellate::flatten_quad(
                    PointF::new(last_p.0, last_p.1),
                    *cp,
                    *p,
                    0.25,
                    &mut points,
                );
                for pt in points {
                    add_edge(&mut edges, last_p, (pt.x, pt.y), transform, scale);
                    last_p = (pt.x, pt.y);
                }
            }
            PathVerb::CubicTo(cp1, cp2, p) => {
                let mut points = Vec::new();
                crate::tessellate::flatten_cubic(
                    PointF::new(last_p.0, last_p.1),
                    *cp1,
                    *cp2,
                    *p,
                    0.25,
                    &mut points,
                );
                for pt in points {
                    add_edge(&mut edges, last_p, (pt.x, pt.y), transform, scale);
                    last_p = (pt.x, pt.y);
                }
            }
            PathVerb::Close => {
                if last_p != start_p {
                    add_edge(&mut edges, last_p, start_p, transform, scale);
                }
                last_p = start_p;
            }
            _ => {} // Quad/Cubic would need flattening
        }
    }
    edges
}

fn add_edge(
    edges: &mut Vec<Edge>,
    p1: (f32, f32),
    p2: (f32, f32),
    transform: &Transform2D,
    scale: i32,
) {
    let (x1, y1) = transform.transform_point_f(p1.0, p1.1);
    let (x2, y2) = transform.transform_point_f(p2.0, p2.1);

    let f_scale = scale as f32;
    let (x1, y1) = (x1 * f_scale, y1 * f_scale);
    let (x2, y2) = (x2 * f_scale, y2 * f_scale);

    if y1 == y2 {
        return;
    }

    let (p1, p2, winding) = if y1 < y2 {
        ((x1, y1), (x2, y2), 1)
    } else {
        ((x2, y2), (x1, y1), -1)
    };

    let dx_dy = ((p2.0 - p1.0) / (p2.1 - p1.1) * 65536.0) as i32;
    edges.push(Edge {
        x: (p1.0 * 65536.0) as i32,
        dx_dy,
        y_min: p1.1 as i32,
        y_max: p2.1 as i32,
        winding,
    });
}

fn blend(dst: u32, src: u32) -> u32 {
    let sa = (src >> 24) & 0xFF;
    if sa == 255 {
        return src;
    }
    if sa == 0 {
        return dst;
    }

    let inv_sa = 255 - sa;
    let sr = (src >> 16) & 0xFF;
    let sg = (src >> 8) & 0xFF;
    let sb = src & 0xFF;

    let dr = (dst >> 16) & 0xFF;
    let dg = (dst >> 8) & 0xFF;
    let db = dst & 0xFF;

    let r = (sr * sa + dr * inv_sa) / 255;
    let g = (sg * sa + dg * inv_sa) / 255;
    let b = (sb * sa + db * inv_sa) / 255;

    (0xFF << 24) | (r << 16) | (g << 8) | b
}
