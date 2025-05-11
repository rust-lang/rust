//@ known-bug: #135863
struct A;

impl A {
    fn len(self: &&B) {}
}

fn main() {
    A.len()
}
