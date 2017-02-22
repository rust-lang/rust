// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A regression test extracted from image-0.3.11. The point of
// failure was in `index_colors` below.

#![allow(unused)]

use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone)]
pub struct Luma<T: Primitive> { pub data: [T; 1] }

impl<T: Primitive + 'static> Pixel for Luma<T> {
    type Subpixel = T;
}

pub struct ImageBuffer<P: Pixel, Container> {
    pixels: P,
    c: Container,
}

pub trait GenericImage: Sized {
    type Pixel: Pixel;
}

pub trait Pixel: Copy + Clone {
    type Subpixel: Primitive;
}

pub trait Primitive: Copy + PartialOrd<Self> + Clone  {
}

impl<P, Container> GenericImage for ImageBuffer<P, Container>
where P: Pixel + 'static,
      Container: Deref<Target=[P::Subpixel]> + DerefMut,
      P::Subpixel: 'static {

    type Pixel = P;
}

impl Primitive for u8 { }

impl<P, Container> ImageBuffer<P, Container>
where P: Pixel + 'static,
      P::Subpixel: 'static,
      Container: Deref<Target=[P::Subpixel]>
{
    pub fn pixels<'a>(&'a self) -> Pixels<'a, Self> {
        loop { }
    }

    pub fn pixels_mut(&mut self) -> PixelsMut<P> {
        loop { }
    }
}

pub struct Pixels<'a, I: 'a> {
    image:  &'a I,
    x:      u32,
    y:      u32,
    width:  u32,
    height: u32
}

impl<'a, I: GenericImage> Iterator for Pixels<'a, I> {
    type Item = (u32, u32, I::Pixel);

    fn next(&mut self) -> Option<(u32, u32, I::Pixel)> {
        loop { }
    }
}

pub struct PixelsMut<'a, P: Pixel + 'a> where P::Subpixel: 'a {
    chunks: &'a mut P::Subpixel
}

impl<'a, P: Pixel + 'a> Iterator for PixelsMut<'a, P> where P::Subpixel: 'a {
    type Item = &'a mut P;

    fn next(&mut self) -> Option<&'a mut P> {
        loop { }
    }
}

pub fn index_colors<Pix>(image: &ImageBuffer<Pix, Vec<u8>>)
                         -> ImageBuffer<Luma<u8>, Vec<u8>>
where Pix: Pixel<Subpixel=u8> + 'static,
{
    let mut indices: ImageBuffer<Luma<u8>, Vec<u8>> = loop { };
    for (pixel, idx) in image.pixels().zip(indices.pixels_mut()) {
        // failured occurred here ^^ because we were requiring that we
        // could project Pixel or Subpixel from `T_indices` (type of
        // `indices`), but the type is insufficiently constrained
        // until we reach the return below.
    }
    indices
}

fn main() { }
