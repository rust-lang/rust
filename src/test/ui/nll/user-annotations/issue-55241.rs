// Regression test for #55241:
//
// The reference to `C::HASHED_NULL_NODE` resulted in a type like `<C
// as NodeCodec<_>>::Out`; normalizing this type requires knowing the
// value of `_`; solving that requires having normalized, so we can
// test against `C: NodeCodec<H>` in the environment.
//
// run-pass

pub trait Hasher {
    type Out: Eq;
}

pub trait NodeCodec<H: Hasher> {
    const HASHED_NULL_NODE: H::Out;
}

pub trait Trie<H: Hasher, C: NodeCodec<H>> {
    /// Returns the root of the trie.
    fn root(&self) -> &H::Out;

    /// Is the trie empty?
    fn is_empty(&self) -> bool { *self.root() == C::HASHED_NULL_NODE }
}

fn main() { }
