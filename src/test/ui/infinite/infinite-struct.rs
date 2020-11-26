struct Take(Take);
//~^ ERROR has infinite size
//~| ERROR cycle detected

// check that we don't hang trying to find the tail of a recursive struct (#79437)
fn foo() -> Take {
    Take(loop {})
}

fn main() {}
