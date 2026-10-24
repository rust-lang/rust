fn main() {
    let mut v: Vec<()> = Vec::new();
    || &mut v;
//~^ ERROR captured variable cannot escape `FnMut` closure body
}
