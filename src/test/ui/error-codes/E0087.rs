fn foo() {}
fn bar<T>() {}

fn main() {
    foo::<f64>(); //~ ERROR wrong number of type arguments: expected 0, found 1 [E0087]

    bar::<f64, u64>(); //~ ERROR wrong number of type arguments: expected 1, found 2 [E0087]
}
