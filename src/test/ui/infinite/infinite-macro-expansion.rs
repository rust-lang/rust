macro_rules! recursive {
    () => (recursive!()) //~ ERROR recursion limit reached while expanding the macro `recursive`
}

fn main() {
    recursive!()
}
