// pp-exact
// Testing that both the inner item and next outer item are
// preserved, and that the first outer item parsed in main is not
// accidentally carried over to each inner function

fn main() {
    #[inner_attr];
    #[outer_attr]
    fn f() { }

    #[outer_attr]
    fn g() { }
}
