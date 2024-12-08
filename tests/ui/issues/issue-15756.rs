use std::slice::Chunks;
use std::slice::ChunksMut;

fn dft_iter<'a, T>(arg1: Chunks<'a,T>, arg2: ChunksMut<'a,T>)
{
    for
    &mut something
    //~^ ERROR the size for values of type
    in arg2
    {
    }
}

fn main() {}
