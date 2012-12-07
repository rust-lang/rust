// xfail-test
fn foo<T, U>(x: T, y: U) {
    let mut xx = x;
    xx = y; // error message should mention T and U, not 'a and 'b
}

fn main() {
    
}