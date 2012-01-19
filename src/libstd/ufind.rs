
import core::option;
import option::{some, none};


// A very naive implementation of union-find with unsigned integer nodes.
// Maintains the invariant that the root of a node is always equal to or less
// than the node itself.
type node = option::t<uint>;

type ufind = {mutable nodes: [mutable node]};

fn make() -> ufind { ret {mutable nodes: [mutable]}; }

fn make_set(ufnd: ufind) -> uint {
    let idx = vec::len(ufnd.nodes);
    ufnd.nodes += [mutable none::<uint>];
    ret idx;
}


/// Creates sets as necessary to ensure that least `n` sets are present in the
/// data structure.
fn grow(ufnd: ufind, n: uint) {
    while set_count(ufnd) < n { make_set(ufnd); }
}

fn find(ufnd: ufind, n: uint) -> uint {
    alt ufnd.nodes[n] {
      none { ret n; }
      some(m) { let m_ = m; be find(ufnd, m_); }
    }
}

fn union(ufnd: ufind, m: uint, n: uint) {
    let m_root = find(ufnd, m);
    let n_root = find(ufnd, n);
    if m_root < n_root {
        ufnd.nodes[n_root] = some::<uint>(m_root);
    } else if m_root > n_root { ufnd.nodes[m_root] = some::<uint>(n_root); }
}

fn set_count(ufnd: ufind) -> uint { ret vec::len::<node>(ufnd.nodes); }


// Removes all sets with IDs greater than or equal to the given value.
fn prune(ufnd: ufind, n: uint) {
    // TODO: Use "slice" once we get rid of "const"

    let len = vec::len::<node>(ufnd.nodes);
    while len != n { vec::pop::<node>(ufnd.nodes); len -= 1u; }
}
