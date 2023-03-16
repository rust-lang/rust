//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-ignore-leaks

fn main() {
    std::mem::forget(Box::new(42));
}
