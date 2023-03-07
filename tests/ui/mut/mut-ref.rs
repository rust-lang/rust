fn main() {
    let mut ref x = 10; //~ ERROR the order of `mut` and `ref` is incorrect
    let ref mut y = 11;
}
