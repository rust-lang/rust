Pattern arm did not match expected kind.

Erroneous code example:

```compile_fail,E0532
enum State {
    Succeeded,
    Failed(String),
}

fn print_on_failure(state: &State) {
    match *state {
        // error: expected unit struct, unit variant or constant, found tuple
        //        variant `State::Failed`
        State::Failed => println!("Failed"),
        _ => ()
    }
}
```

To fix this error, ensure the match arm kind is the same as the expression
matched.

Fixed example:

```
enum State {
    Succeeded,
    Failed(String),
}

fn print_on_failure(state: &State) {
    match *state {
        State::Failed(ref msg) => println!("Failed with {}", msg),
        _ => ()
    }
}
```
