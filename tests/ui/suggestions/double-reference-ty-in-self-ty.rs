// issue#135863

struct A;

impl A {
    fn len(self: &&A) {}
}

fn main() {
    A.len();
    //~^ ERROR: no method named `len` found for struct `A` in the current scope
}
