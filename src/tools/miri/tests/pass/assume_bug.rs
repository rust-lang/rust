//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
fn main() {
    vec![()].into_iter();
}
