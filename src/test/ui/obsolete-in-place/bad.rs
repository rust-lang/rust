// Check that `<-` and `in` syntax gets a hard error.

// revisions: good bad
//[good] run-pass

#[cfg(bad)]
fn main() {
    let (x, y, foo, bar);
    x <- y; //[bad]~ ERROR emplacement syntax is obsolete
    in(foo) { bar }; //[bad]~ ERROR emplacement syntax is obsolete
}

#[cfg(good)]
fn main() {
}
