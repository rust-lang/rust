fn main() {
    let zero = assert_eq::<()>();
    //~^ ERROR expected function, found macro `assert_eq`
}
