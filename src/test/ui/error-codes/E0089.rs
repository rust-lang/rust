fn foo<T, U>() {}

fn main() {
    foo::<f64>(); //~ ERROR wrong number of type arguments: expected 2, found 1 [E0089]
}
