trait Owner {
    const C<const N: u32>: u32 = N;
    //~^ ERROR: generic const items are experimental
}

impl Owner for () {}
fn take0<const N: u64>(_: impl Owner<C<N> = { N }>) {}
//~^ ERROR: associated const equality is incomplete

fn main() {
    take0::<128>(());
}
