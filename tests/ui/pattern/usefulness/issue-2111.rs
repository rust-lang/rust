fn foo(a: Option<usize>, b: Option<usize>) {
    match (a, b) {
        //~^ ERROR: match is non-exhaustive
        (Some(a), Some(b)) if a == b => {}
        (Some(_), None) | (None, Some(_)) => {}
    }
}

fn main() {
    foo(None, None);
}
