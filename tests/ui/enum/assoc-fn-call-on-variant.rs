#[derive(Default)]
enum E {
    A {},
    B {},
    #[default]
    C,
}

impl E {
    fn f() {}
}

fn main() {
    E::A::f(); //~ ERROR cannot find module `A` in `E`
    //~^ NOTE: `A` is a variant, not a module
}
