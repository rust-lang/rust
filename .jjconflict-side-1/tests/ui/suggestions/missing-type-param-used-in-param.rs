//@ run-rustfix

fn two_type_params<A, B>(_: B) {}

fn main() {
    two_type_params::<String>(100); //~ ERROR function takes 2 generic arguments
    two_type_params::<String, _>(100);
}
