// Regression test for https://github.com/rust-lang/rust/issues/143047

mod mrow {
    pub struct InBoundsIndex<const N: usize>(());

    impl<const N: usize> InBoundsIndex<N> {
        pub const fn new() -> Option<InBoundsIndex<N>> {
            if N < 32 { Some(Self(())) } else { None }
        }
    }
}

use mrow::InBoundsIndex;

static IDX: InBoundsIndex<64> = {
    // The following line should cause a compilation error, otherwise it results in an
    // undefined behavior.
    index([0; 32], &IDX);
    //~^ ERROR: encountered static that tried to access itself during initialization
    InBoundsIndex::<64>::new().unwrap()
};

const fn index<const N: usize>(arr: [u8; 32], _: &InBoundsIndex<N>) -> u8 {
    // SAFETY: InBoundsIndex can only be created by its new, which ensures N is < 32
    unsafe { arr.as_ptr().add(N).read() }
}

fn main() {}
