//@ check-pass
//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-impl-restriction.rs

extern crate coherence_impl_restriction as upstream;

struct LocalIndex(usize);

impl<T> std::ops::Index<LocalIndex> for [T] {
    type Output = T;

    fn index(&self, index: LocalIndex) -> &Self::Output {
        &self[index.0]
    }
}

// Restricting who can implement a trait does not, by itself,
// change the negative reasoning permitted by coherence.
trait LocalTrait {}
impl<T: upstream::UnmarkedRestricted> LocalTrait for T {}
impl LocalTrait for LocalIndex {}

fn main() {}
