// MIR doesn't generate an error because the assignment isn't reachable. This
// is OK because the test is here to check that the compiler doesn't ICE (cf.
// #5500).

// build-pass (FIXME(62277): could be check-pass?)

struct TrieMapIterator<'a> {
    node: &'a usize
}

fn main() {
    let a = 5;
    let _iter = TrieMapIterator{node: &a};
    _iter.node = &panic!()
}
