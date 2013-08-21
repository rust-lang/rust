// Perlin noise benchmark from https://gist.github.com/1170424

use std::float;
use std::rand::{Rng, RngUtil};
use std::rand;

struct Vec2 {
    x: f32,
    y: f32,
}

#[inline(always)]
fn lerp(a: f32, b: f32, v: f32) -> f32 { a * (1.0 - v) + b * v }

#[inline(always)]
fn smooth(v: f32) -> f32 { v * v * (3.0 - 2.0 * v) }

fn random_gradient<R:Rng>(r: &mut R) -> Vec2 {
    let v = 2.0 * float::consts::pi * r.gen();
    Vec2 {
        x: v.cos() as f32,
        y: v.sin() as f32,
    }
}

fn gradient(orig: Vec2, grad: Vec2, p: Vec2) -> f32 {
    let sp = Vec2 {x: p.x - orig.x, y: p.y - orig.y};
    grad.x * sp.x + grad.y * sp.y
}

struct Noise2DContext {
    rgradients: [Vec2, ..256],
    permutations: [int, ..256],
}

impl Noise2DContext {
    pub fn new() -> Noise2DContext {
        let mut r = rand::rng();
        let mut rgradients = [ Vec2 { x: 0.0, y: 0.0 }, ..256 ];
        for i in range(0, 256) {
            rgradients[i] = random_gradient(&mut r);
        }
        let mut permutations = [ 0, ..256 ];
        for i in range(0, 256) {
            permutations[i] = i;
        }
        r.shuffle_mut(permutations);

        Noise2DContext {
            rgradients: rgradients,
            permutations: permutations,
        }
    }

    #[inline(always)]
    pub fn get_gradient(&self, x: int, y: int) -> Vec2 {
        let idx = self.permutations[x & 255] + self.permutations[y & 255];
        self.rgradients[idx & 255]
    }

    #[inline]
    pub fn get_gradients(&self,
                         gradients: &mut [Vec2, ..4],
                         origins: &mut [Vec2, ..4],
                         x: f32,
                         y: f32) {
        let x0f = x.floor();
        let y0f = y.floor();
        let x0 = x0f as int;
        let y0 = y0f as int;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        gradients[0] = self.get_gradient(x0, y0);
        gradients[1] = self.get_gradient(x1, y0);
        gradients[2] = self.get_gradient(x0, y1);
        gradients[3] = self.get_gradient(x1, y1);

        origins[0] = Vec2 {x: x0f + 0.0, y: y0f + 0.0};
        origins[1] = Vec2 {x: x0f + 1.0, y: y0f + 0.0};
        origins[2] = Vec2 {x: x0f + 0.0, y: y0f + 1.0};
        origins[3] = Vec2 {x: x0f + 1.0, y: y0f + 1.0};
    }

    #[inline]
    pub fn get(&self, x: f32, y: f32) -> f32 {
        let p = Vec2 {x: x, y: y};
        let mut gradients = [ Vec2 { x: 0.0, y: 0.0 }, ..4 ];
        let mut origins = [ Vec2 { x: 0.0, y: 0.0 }, ..4 ];
        self.get_gradients(&mut gradients, &mut origins, x, y);
        let v0 = gradient(origins[0], gradients[0], p);
        let v1 = gradient(origins[1], gradients[1], p);
        let v2 = gradient(origins[2], gradients[2], p);
        let v3 = gradient(origins[3], gradients[3], p);
        let fx = smooth(x - origins[0].x);
        let vx0 = lerp(v0, v1, fx);
        let vx1 = lerp(v2, v3, fx);
        let fy = smooth(y - origins[0].y);
        lerp(vx0, vx1, fy)
    }
}

fn main() {
    let symbols = [" ", "░", "▒", "▓", "█", "█"];
    let mut pixels = [0f32, ..256*256];
    let n2d = ~Noise2DContext::new();
    for _ in range(0, 100u) {
        for y in range(0, 256) {
            for x in range(0, 256) {
                let v = n2d.get(
                    x as f32 * 0.1f32,
                    y as f32 * 0.1f32
                ) * 0.5f32 + 0.5f32;
                pixels[y*256+x] = v;
            };
        };
    };

    for y in range(0, 256) {
        for x in range(0, 256) {
            print(symbols[(pixels[y*256+x] / 0.2f32) as int]);
        }
        println("");
    }
}
