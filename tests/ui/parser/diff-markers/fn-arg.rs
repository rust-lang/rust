trait T {
    fn foo(
<<<<<<< HEAD //~ ERROR encountered git conflict marker
        x: u8,
=======
        x: i8,
>>>>>>> branch
    ) {}
}

struct S;
impl T for S {}

fn main() {
    S::foo(42);
}
