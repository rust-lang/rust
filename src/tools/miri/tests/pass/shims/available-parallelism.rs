//@revisions: stack tree tree_implicit_writes
//@[tree_implicit_writes]compile-flags: -Zmiri-tree-borrows -Zmiri-tree-borrows-implicit-writes
//@[tree]compile-flags: -Zmiri-tree-borrows
fn main() {
    assert_eq!(std::thread::available_parallelism().unwrap().get(), 1);
}
