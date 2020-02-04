use std::num::NonZeroUsize;

fn main() {
    let a: usize = 1.try_into().unwrap();
    #[should_panic] let b: usize = 0.try_into().unwrap();
}
