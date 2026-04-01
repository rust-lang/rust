// This test is a reduced version of a bug introduced during work on type-tests for Polonius.
// The underlying problem is that the 'static bound is lost for a type parameter that is
// threaded deeply enough, causing an error.
// The bug was first observed in exr-1.4.1/src/image/read/mod.rs:124:5 during perf test.

//@ check-pass

use std::marker::PhantomData;

struct ReadAllLayers<ReadChannels> {
    px: PhantomData<ReadChannels>,
}

trait ReadLayers<'s> {}

impl<'s, C> ReadLayers<'s> for ReadAllLayers<C> where C: ReadChannels<'s> {}

fn make_builder<A, Set, Pixels>(
    _: Set,
) -> ReadAllLayers<CollectPixels<A, Pixels, Set>>
where
    Set: Fn(&mut Pixels),
{
    todo!()
}

struct CollectPixels<Pixel, PixelStorage, SetPixel> {
    px: PhantomData<(SetPixel, Pixel, PixelStorage)>,
}

impl<'s, PixelStorage, SetPixel: 's> ReadChannels<'s>
    for CollectPixels<usize, PixelStorage, SetPixel>
where
    SetPixel: Fn(&mut PixelStorage),
{
}

trait ReadChannels<'s> {}

fn from_file<L>(_: L)
where
    for<'s> L: ReadLayers<'s>,
{
}

pub fn read_all_rgba_layers_from_file<Set: 'static, Pixels: 'static>(
    set_pixel: Set,
) where
    Set: Fn(&mut Pixels),
{
    from_file(make_builder(set_pixel)); // Error triggered.
}

pub fn main() {}
