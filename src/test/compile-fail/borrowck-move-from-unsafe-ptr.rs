fn foo(x: *~int) -> ~int {
    let y <- *x; //~ ERROR dereference of unsafe pointer requires unsafe function or block
    return y;
}

fn main() {
}