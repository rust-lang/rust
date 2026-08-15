//@ run-pass

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]
// A regression test extracted from image-0.3.11. The point of
// failure was in `index_colors` below.

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

    pub fn pixels_mut(&mut self) -> PixelsMut<'_, P> {
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
    // When NLL-enabled, `let mut` below is deemed unnecessary (due to
    // the remaining code being unreachable); so ignore that lint.
    #![allow(unused_mut)]

    let mut indices: ImageBuffer<_,Vec<_>> = loop { };
    for (pixel, idx) in image.pixels().zip(indices.pixels_mut()) {
        // failure occurred here ^^ because we were requiring that we
        // could project Pixel or Subpixel from `T_indices` (type of
        // `indices`), but the type is insufficiently constrained
        // until we reach the return below.
    }
    indices
}

fn main() { }
