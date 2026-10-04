//@ check-pass
//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally

struct SmallIndex(usize);

impl<T> core::ops::Index<SmallIndex> for [T] {
    type Output = T;

    fn index(&self, index: SmallIndex) -> &Self::Output {
        &self[index.0]
    }
}

fn main() {}
