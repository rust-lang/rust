macro_rules! recursive {
    () => (recursive!()) //~ ERROR recursion limit reached while expanding `recursive!`
}

fn main() {
    recursive!()
}
