trait T {
<<<<<<< HEAD //~ ERROR encountered git conflict marker
    fn foo() {}
=======
    fn bar() {}
>>>>>>> branch
}

struct S;
impl T for S {}

fn main() {
    S::foo();
}
