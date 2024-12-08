trait Extend {
    fn extend(_: &str) -> (impl Sized + '_, &'static str);
}

impl Extend for () {
    fn extend(s: &str) -> (Option<&'static &'_ ()>, &'static str) {
        //~^ ERROR in type `&'static &()`, reference has a longer lifetime than the data it references
        (None, s)
    }
}

// This indirection is not necessary for reproduction,
// but it makes this test future-proof against #114936.
fn extend<T: Extend>(s: &str) -> &'static str {
    <T as Extend>::extend(s).1
}

fn main() {
    let use_after_free = extend::<()>(&String::from("temporary"));
    println!("{}", use_after_free);
}
