//@ check-pass

fn main() {
    // Common
    for _ in Some(1) {}
    //~^ WARN for loop over an `Option`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent
    for _ in Ok::<_, ()>(1) {}
    //~^ WARN for loop over a `Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent

    // `Iterator::next` specific
    for _ in [0; 0].iter().next() {}
    //~^ WARN for loop over an `Option`. This is more readably written as an `if let` statement
    //~| HELP to iterate over `[0; 0].iter()` remove the call to `next`
    //~| HELP consider using `if let` to clear intent

    // `Result<impl Iterator, _>`, but function doesn't return `Result`
    for _ in Ok::<_, ()>([0; 0].iter()) {}
    //~^ WARN for loop over a `Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent
}

fn _returns_result() -> Result<(), ()> {
    // `Result<impl Iterator, _>`
    for _ in Ok::<_, ()>([0; 0].iter()) {}
    //~^ WARN for loop over a `Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider unwrapping the `Result` with `?` to iterate over its contents
    //~| HELP consider using `if let` to clear intent

    // `Result<impl IntoIterator>`
    for _ in Ok::<_, ()>([0; 0]) {}
    //~^ WARN for loop over a `Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider unwrapping the `Result` with `?` to iterate over its contents
    //~| HELP consider using `if let` to clear intent

    Ok(())
}

fn _by_ref() {
    // Shared refs
    for _ in &Some(1) {}
    //~^ WARN for loop over a `&Option`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent
    for _ in &Ok::<_, ()>(1) {}
    //~^ WARN for loop over a `&Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent

    // Mutable refs
    for _ in &mut Some(1) {}
    //~^ WARN for loop over a `&mut Option`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent
    for _ in &mut Ok::<_, ()>(1) {}
    //~^ WARN for loop over a `&mut Result`. This is more readably written as an `if let` statement
    //~| HELP to check pattern in a loop use `while let`
    //~| HELP consider using `if let` to clear intent
}
