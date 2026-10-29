//@ check-pass
//@ compile-flags: -Znext-solver=globally

pub struct IndexMap<S, const N: usize> {
    #[expect(dead_code)]
    build_hasher: S,
}

struct Guard<'a, S, const N: usize>(
    #[expect(dead_code)] &'a mut IndexMap<S, N>,
);

impl<S, const N: usize> IndexMap<S, N> {
    pub fn clear(&mut self) {
        let _ = Guard(self);
    }
}

fn main() {}
