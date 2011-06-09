import option::none;
import option::some;

// A very naive implementation of union-find with unsigned integer nodes.
// Maintains the invariant that the root of a node is always equal to or less
// than the node itself.

type node = option::t[uint];
type ufind = rec(mutable vec[mutable node] nodes);

fn make() -> ufind {
    let vec[mutable node] v = [mutable none[uint]];
    vec::pop(v);  // FIXME: botch
    ret rec(mutable nodes=v);
}

fn make_set(&ufind ufnd) -> uint {
    auto idx = vec::len(ufnd.nodes);
    ufnd.nodes += [mutable none[uint]];
    ret idx;
}

/// Creates sets as necessary to ensure that least `n` sets are present in the
/// data structure.
fn grow(&ufind ufnd, uint n) {
    while (set_count(ufnd) < n) { make_set(ufnd); }
}

fn find(&ufind ufnd, uint n) -> uint {
    alt (ufnd.nodes.(n)) {
        case (none) { ret n; }
        case (some(?m)) { auto m_ = m; be find(ufnd, m_); }
    }
}

fn union(&ufind ufnd, uint m, uint n) {
    auto m_root = find(ufnd, m);
    auto n_root = find(ufnd, n);
    if (m_root < n_root) {
        ufnd.nodes.(n_root) = some[uint](m_root);
    } else if (m_root > n_root) {
        ufnd.nodes.(m_root) = some[uint](n_root);
    }
}

fn set_count(&ufind ufnd) -> uint {
    ret vec::len[node](ufnd.nodes);
}

// Removes all sets with IDs greater than or equal to the given value.
fn prune(&ufind ufnd, uint n) {
    // TODO: Use "slice" once we get rid of "mutable?"
    auto len = vec::len[node](ufnd.nodes);
    while (len != n) {
        vec::pop[node](ufnd.nodes);
        len -= 1u;
    }
}

