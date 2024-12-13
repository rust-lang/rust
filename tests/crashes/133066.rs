//@ known-bug: #133066
trait Owner {
    const C<const N: u32>: u32;
}

impl Owner for () {;}

fn take0<const N: u64>(_: impl Owner<C<N> = { N }>) {}

fn main() {
    take0::<f32, >(());
}
