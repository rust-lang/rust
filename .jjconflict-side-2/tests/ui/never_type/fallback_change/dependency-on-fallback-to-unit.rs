fn main() {
    def();
    _ = question_mark();
}

fn def() {
    match true {
        false => <_>::default(),
        //~^ error: the trait bound `!: Default` is not satisfied
        true => return,
    };
}

// <https://github.com/rust-lang/rust/issues/51125>
// <https://github.com/rust-lang/rust/issues/39216>
fn question_mark() -> Result<(), ()> {
    deserialize()?;
    //~^ error: the trait bound `!: Default` is not satisfied
    Ok(())
}

fn deserialize<T: Default>() -> Result<T, ()> {
    Ok(T::default())
}
