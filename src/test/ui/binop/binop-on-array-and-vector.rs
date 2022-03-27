fn main() {
    struct X;
    let foo = [X, X, X];
    foo == [X, X, X]; //~ ERROR binary operation `==` cannot be applied to type `[X; 3]` [E0369]

    let bar = vec![X, X, X];
    bar == [X, X, X]; //~ ERROR binary operation `==` cannot be applied to type `Vec<X>` [E0369]
}
