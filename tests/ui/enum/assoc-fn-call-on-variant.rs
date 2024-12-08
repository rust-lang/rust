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
    E::A::f(); //~ ERROR failed to resolve: `A` is a variant, not a module
}
