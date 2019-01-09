// ignore-compare-mode-nll
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=compare

struct TrieMapIterator<'a> {
    node: &'a usize
}

fn main() {
    let a = 5;
    let _iter = TrieMapIterator{node: &a};
    _iter.node = & //[ast]~ ERROR cannot assign to field `_iter.node` of immutable binding
                   //[mir]~^ ERROR cannot assign to field `_iter.node` of immutable binding (Ast)
                   // MIR doesn't generate an error because the code isn't reachable. This is OK
                   // because the test is here to check that the compiler doesn't ICE (cf. #5500).
    panic!()
}
