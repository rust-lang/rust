#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::geometry::{Color, LineCap, LineJoin, PointF};
use alloc::vec::Vec;

/// Flatten a quadratic bezier curve into line segments using adaptive subdivision.
pub fn flatten_quad(p0: PointF, cp: PointF, p1: PointF, tolerance: f32, output: &mut Vec<PointF>) {
    flatten_quad_recursive(p0, cp, p1, tolerance, output, 0);
}

fn flatten_quad_recursive(
    p0: PointF,
    cp: PointF,
    p1: PointF,
    tolerance: f32,
    output: &mut Vec<PointF>,
    depth: u32,
) {
    const MAX_DEPTH: u32 = 16;
    if depth > MAX_DEPTH {
        output.push(p1);
        return;
    }

    let t = 0.5;
    let mid = eval_quad(p0, cp, p1, t);
    let chord_mid = PointF::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5);

    let dx = mid.x - chord_mid.x;
    let dy = mid.y - chord_mid.y;
    let dist_sq = dx * dx + dy * dy;

    if dist_sq <= tolerance * tolerance {
        output.push(p1);
    } else {
        let cp0 = PointF::new((p0.x + cp.x) * 0.5, (p0.y + cp.y) * 0.5);
        let cp1 = PointF::new((cp.x + p1.x) * 0.5, (cp.y + p1.y) * 0.5);
        flatten_quad_recursive(p0, cp0, mid, tolerance, output, depth + 1);
        flatten_quad_recursive(mid, cp1, p1, tolerance, output, depth + 1);
    }
}

fn eval_quad(p0: PointF, cp: PointF, p1: PointF, t: f32) -> PointF {
    let t2 = 1.0 - t;
    let b0 = t2 * t2;
    let b1 = 2.0 * t2 * t;
    let b2 = t * t;
    PointF::new(
        b0 * p0.x + b1 * cp.x + b2 * p1.x,
        b0 * p0.y + b1 * cp.y + b2 * p1.y,
    )
}

/// Flatten a cubic bezier curve into line segments.
pub fn flatten_cubic(
    p0: PointF,
    cp1: PointF,
    cp2: PointF,
    p1: PointF,
    tolerance: f32,
    output: &mut Vec<PointF>,
) {
    flatten_cubic_recursive(p0, cp1, cp2, p1, tolerance, output, 0);
}

fn flatten_cubic_recursive(
    p0: PointF,
    cp1: PointF,
    cp2: PointF,
    p1: PointF,
    tolerance: f32,
    output: &mut Vec<PointF>,
    depth: u32,
) {
    const MAX_DEPTH: u32 = 16;
    if depth > MAX_DEPTH {
        output.push(p1);
        return;
    }

    let mid = eval_cubic(p0, cp1, cp2, p1, 0.5);
    let chord_mid = PointF::new((p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5);

    let dx = mid.x - chord_mid.x;
    let dy = mid.y - chord_mid.y;
    let dist_sq = dx * dx + dy * dy;

    if dist_sq <= tolerance * tolerance {
        output.push(p1);
    } else {
        let q0 = lerp_point(p0, cp1, 0.5);
        let q1 = lerp_point(cp1, cp2, 0.5);
        let q2 = lerp_point(cp2, p1, 0.5);
        let r0 = lerp_point(q0, q1, 0.5);
        let r1 = lerp_point(q1, q2, 0.5);
        let split = lerp_point(r0, r1, 0.5);

        flatten_cubic_recursive(p0, q0, r0, split, tolerance, output, depth + 1);
        flatten_cubic_recursive(split, r1, q2, p1, tolerance, output, depth + 1);
    }
}

fn eval_cubic(p0: PointF, cp1: PointF, cp2: PointF, p1: PointF, t: f32) -> PointF {
    let t2 = 1.0 - t;
    let b0 = t2 * t2 * t2;
    let b1 = 3.0 * t2 * t2 * t;
    let b2 = 3.0 * t2 * t * t;
    let b3 = t * t * t;
    PointF::new(
        b0 * p0.x + b1 * cp1.x + b2 * cp2.x + b3 * p1.x,
        b0 * p0.y + b1 * cp1.y + b2 * cp2.y + b3 * p1.y,
    )
}

fn lerp_point(p0: PointF, p1: PointF, t: f32) -> PointF {
    PointF::new(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t)
}

// ── Stroke Expansion ──────────────────────────────────────────────────────────

pub struct StrokeStyle {
    pub width: f32,
    pub line_cap: LineCap,
    pub line_join: LineJoin,
    pub miter_limit: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            width: 1.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 4.0,
        }
    }
}

pub struct TessellatedPath {
    pub vertices: Vec<PointF>,
    pub contours: Vec<Contour>,
}

pub struct Contour {
    pub start: usize,
    pub count: usize,
}

impl TessellatedPath {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            contours: Vec::new(),
        }
    }
}

pub fn expand_stroke(path: &TessellatedPath, style: &StrokeStyle) -> TessellatedPath {
    let mut result = TessellatedPath::new();
    let offset = style.width * 0.5;

    for contour in &path.contours {
        if contour.count < 2 {
            continue;
        }

        let start_idx = result.vertices.len();
        for i in 0..contour.count {
            let curr = path.vertices[contour.start + i];

            // Calculate normal
            let (px, py) = if i + 1 < contour.count {
                let next = path.vertices[contour.start + i + 1];
                let dx = next.x - curr.x;
                let dy = next.y - curr.y;
                let len = libm::sqrtf(dx * dx + dy * dy).max(0.001);
                (-dy / len * offset, dx / len * offset)
            } else {
                let prev = path.vertices[contour.start + i - 1];
                let dx = curr.x - prev.x;
                let dy = curr.y - prev.y;
                let len = libm::sqrtf(dx * dx + dy * dy).max(0.001);
                (-dy / len * offset, dx / len * offset)
            };

            result.vertices.push(PointF::new(curr.x + px, curr.y + py));
            result.vertices.push(PointF::new(curr.x - px, curr.y - py));
        }
        result.contours.push(Contour {
            start: start_idx,
            count: result.vertices.len() - start_idx,
        });
    }

    result
}
