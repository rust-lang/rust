//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-ignore-leaks

fn main() {
    std::mem::forget(Box::new(42));
}
