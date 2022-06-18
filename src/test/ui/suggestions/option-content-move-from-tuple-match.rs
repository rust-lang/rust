fn foo(a: &Option<String>, b: &Option<String>) {
    match (a, b) {
        //~^ ERROR cannot move out of a shared reference
        (None, &c) => &c.unwrap(),
        (&Some(ref c), _) => c,
    };
}

fn main() {}
