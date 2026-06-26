// Regression test making sure that indexing fails with an ambiguity
// error if one of the deref-steps encounters an inference variable.

fn main() {
    let x = &Default::default();
    //~^ ERROR type annotations needed for `&_`
    x[1];
    let _: &Vec<()> = x;
}
