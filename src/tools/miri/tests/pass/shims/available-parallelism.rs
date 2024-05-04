//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
fn main() {
    assert_eq!(std::thread::available_parallelism().unwrap().get(), 1);
}
