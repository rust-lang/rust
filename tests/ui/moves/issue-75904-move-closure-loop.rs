// Regression test for issue #75904
// Tests that we point at an expression
// that required the upvar to be moved, rather than just borrowed.

struct NotCopy;

fn main() {
    let mut a = NotCopy;
    loop {
        || { //~ ERROR use of moved value
            &mut a;
            a;
        };
    }
}
