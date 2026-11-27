fn main() {
}

fn foo<T>(x: T, y: T) {
    let z = x + y; //~ ERROR cannot add `T` to `T`
}

fn bar<T>(x: T) {
    x += x; //~ ERROR binary assignment operation `+=` cannot be applied to type `T`
}

fn baz<T>(x: T) {
    let y = -x; //~ ERROR cannot apply unary operator `-` to type `T`
    let y = !x; //~ ERROR cannot apply unary operator `!` to type `T`
    let y = *x; //~ ERROR type `T` cannot be dereferenced
}
