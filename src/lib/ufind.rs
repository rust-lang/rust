import option::none;
import option::some;

// A very naive implementation of union-find with unsigned integer nodes.

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

fn find(&ufind ufnd, uint n) -> uint {
    alt (ufnd.nodes.(n)) {
    case (none[uint]) { ret n; }
    case (some[uint](?m)) {
        // TODO: "be"
        ret find(ufnd, m);
    }
    }
}

fn union(&ufind ufnd, uint m, uint n) {
    auto m_root = find(ufnd, m);
    auto n_root = find(ufnd, n);
    auto ptr = some[uint](n_root);
    ufnd.nodes.(m_root) = ptr;
}

