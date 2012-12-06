fn a_val(&&x: ~int, +y: ~int) -> int {
    *x + *y
}

fn main() {
    let z = ~22;
    a_val(copy z, copy z);
}
