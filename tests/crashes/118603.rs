//@ known-bug: #118603
//@ compile-flags: -Copt-level=0
// ignore-tidy-linelength

#![feature(generic_const_exprs)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct FlatTree;

#[derive(Copy, Clone)]
struct TreeLeaf;

#[derive(Copy, Clone)]
struct TreeNode<V, W>(V, W);

const fn const_concat<const A: usize, const B: usize>(_: [FlatTree; A], _: [FlatTree; B]) -> [FlatTree; A + B] {
    [FlatTree; A + B]
}

struct Builder<const N: usize, I> {
    ops: [FlatTree; N],
    builder: I,
}

fn create_node<const N: usize, const M: usize, A, B>(a: Builder<N, A>, b: Builder<M, B>) -> Builder<{ N + M + 1 }, TreeNode<A, B>> {
    Builder {
        ops: const_concat(const_concat::<N, M>(a.ops, b.ops), [FlatTree]),
        builder: TreeNode(a.builder, b.builder),
    }
}

const LEAF: Builder<1, TreeLeaf> = Builder {
    ops: [FlatTree],
    builder: TreeLeaf,
};

static INTERNAL_SIMPLE_BOOLEAN_TEMPLATES: &[fn()] = &[{
    fn eval() {
        create_node(LEAF, create_node(LEAF, LEAF));
    }

    eval
}];

pub fn main() {}
