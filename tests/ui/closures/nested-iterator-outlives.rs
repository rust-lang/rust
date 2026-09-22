//@ check-pass
//@ revisions: current coherence next assumptions
//@[current] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

use std::fmt;

struct DebugMap<F>(F);

impl<F, I, K, V> fmt::Debug for DebugMap<F>
where
    F: Fn() -> I,
    I: IntoIterator<Item = (K, V)>,
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries((self.0)()).finish()
    }
}

fn display<T: fmt::Debug>(values: &[(T,)], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Values")
        .field("values", &DebugMap(|| values.iter().map(|value| &value.0).enumerate()))
        .finish()
}

// The iterator stores the closure, not its results. A result type does not
// need to outlive the iterator that produces it (as in petgraph's path iterator).
fn paths<'a, T: std::iter::FromIterator<u8>>() -> impl Iterator<Item = T> + 'a {
    std::iter::from_fn(|| Some(std::iter::empty::<u8>().collect()))
}

fn main() {}
