//@ check-pass

fn main() {
    def();
    _ = question_mark();
}

fn def() {
    //~^ warn: this function depends on never type fallback being `()`
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    match true {
        false => <_>::default(),
        true => return,
    };
}

// <https://github.com/rust-lang/rust/issues/51125>
// <https://github.com/rust-lang/rust/issues/39216>
fn question_mark() -> Result<(), ()> {
    //~^ warn: this function depends on never type fallback being `()`
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    deserialize()?;
    Ok(())
}

fn deserialize<T: Default>() -> Result<T, ()> {
    Ok(T::default())
}
