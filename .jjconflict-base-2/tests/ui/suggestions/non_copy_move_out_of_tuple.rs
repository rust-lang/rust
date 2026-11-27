// Regression test for #146537.

struct NonCopy;
fn main() {
    let tuple = &(NonCopy,);
    let b: NonCopy;
    (b,) = *tuple; //~ ERROR: cannot move out of `tuple.0` which is behind a shared reference [E0507]
}
