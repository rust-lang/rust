#![deny(map_unit_fn)]

#![crate_type = "lib"]
fn _y() {
    vec![42].iter().map(drop);
    //~^ ERROR `Iterator::map` call that discard the iterator's values
}
