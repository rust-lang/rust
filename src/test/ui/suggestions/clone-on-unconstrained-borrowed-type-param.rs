// run-rustfix
fn wat<T>(t: &T) -> T {
    t.clone() //~ ERROR E0308
}

fn main() {
    wat(&42);
}
